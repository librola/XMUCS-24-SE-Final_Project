import argparse
import json
import os

from collections import Counter, OrderedDict
from pathlib import Path

from tqdm import tqdm

from aglibro.util.postprocess_data import extract_python_blocks, normalize_patch
from aglibro.util.utils import load_json, load_jsonl

from aglibro.feedback.test import run_test

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import traceback
from difflib import unified_diff
import docker
import logging
import copy

from datasets import load_dataset
from tqdm import tqdm
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from docker.models.containers import Container

from aglibro.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    # setup_logger,
)
from aglibro.util.model import (
    make_model,
    get_model_price
)
from aglibro.util.api_requests import num_tokens_from_messages
from aglibro.util.postprocess_data import (
    check_code_differ_by_just_empty_lines,
    check_syntax,
    extract_python_blocks,
    fake_git_repo,
    lint_code,
    parse_diff_edit_commands,
    parse_edit_commands,
    remove_empty_lines,
    split_edit_multifile_commands,
)
from aglibro.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
    line_wrap_content,
    transfer_arb_locs_to_locs,
)
from aglibro.util.postprocess_tests import (
    # make_test_script,
    parse_output,
    get_logs_eval,
    get_logs_eval_with_repo,
    MAP_REPO_TO_TEST_PATH,
)
from aglibro.docker.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
)
from aglibro.docker.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
    INSTANCE_IMAGE_BUILD_DIR,
)
from swebench.harness.constants import (
    MAP_REPO_VERSION_TO_SPECS,
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool
from swebench.harness.run_evaluation import EvaluationError, get_dataset_from_preds

from aglibro.libro.llm_prompt import generate_tests
from aglibro.libro.postprocess import run_generate_test
from aglibro.libro.llm_regenerate import regenerate_tests

from aglibro.feedback.test import make_test_script, run_test
from aglibro.repair.repair import construct_topn_file_context, _post_process_multifile_repair, post_process_raw_output
from aglibro.util.postprocess_data import extract_python_blocks, normalize_patch

from aglibro.feedback.rerank import (
    normalize_patches,
    _load_results,
    get_sample,
    get_instance_image_key,
    execution_results,
)

def parse_and_run_instance(instance, test_spec, args, existing_instance_ids):
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    version = instance["version"]
    
    instance_dir = Path(args.output_folder) / "instance_logs" / str(instance_id)
    instance["instance_dir"] = instance_dir
    log_file = instance_dir / f"{instance_id}.log"
    logger = setup_logger(instance_id, log_file, mode="a")
    logger.info(f"Processing instance {instance_id}")
    
    if args.skip_existing and existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Instance {instance_id} already exists in {args.output_file}, skipping.")
        return
    
    if len(execution_results[instance_id]) < args.num_samples:
        print(
            f"There were only {len(execution_results[instance_id])} patches for {instance_id} instead of the full {args.num_samples}"
        )

    patch_keys = [
        execution_results[instance_id][i]["normalized_patch"]
        for i in range(len(execution_results[instance_id]))
    ]
    plausible = [
        execution_results[instance_id][i]["plausible"]
        for i in range(len(execution_results[instance_id]))
    ]
    raw_patches = [
        execution_results[instance_id][i]["patch"]
        for i in range(len(execution_results[instance_id]))
    ]

    if args.plausible:
        patch_ids = [
            i
            for i in range(len(execution_results[instance_id]))
            if patch_keys[i].strip() and plausible[i]
        ]
    else:
        patch_ids = [
            i
            for i in range(len(execution_results[instance_id]))
            if patch_keys[i].strip()
        ]

    if not patch_ids:
        # just vote on all patches
        if not all([x.strip() == "" for x in raw_patches]):
            vote = Counter()
            first_appear_idx = dict()
            valid_indices = []
            for i in range(len(execution_results[instance_id])):
                sample = get_sample(instance_id, i)
                patch_key = sample["normalized_patch"]
                if patch_key != "":
                    valid_indices.append(i)
                    vote[patch_key] += 1
                    if patch_key not in first_appear_idx:
                        first_appear_idx[patch_key] = i
            maj_selected_id = max(
                valid_indices,
                key=lambda i: (
                    vote[patch_keys[i]],
                    -first_appear_idx[patch_keys[i]],
                ),
            )
            
            weighted_indices = sorted(valid_indices, key=lambda i: (
                vote[patch_keys[i]],
                -first_appear_idx[patch_keys[i]],
            ), reverse=True)
        else:
            print(f"No raw patches valid for {instance_id}")
            weighted_indices = []
    else:
        vote = Counter()
        first_appear_idx = dict()
        for i in patch_ids:
            sample = get_sample(instance_id, i)
            patch_key, patch = sample["normalized_patch"], sample["patch"]
            vote[patch_key] += 1
            if patch_key not in first_appear_idx:
                first_appear_idx[patch_key] = i

        maj_selected_id = max(
            patch_ids,
            key=lambda i: (vote[patch_keys[i]], -first_appear_idx[patch_keys[i]]),
        )
        
        weighted_indices = sorted(patch_ids, key=lambda i: (
            vote[patch_keys[i]],
            -first_appear_idx[patch_keys[i]],
        ), reverse=True)

    last_normilized_patch = None
    all_patches = []
    for i, index in enumerate(weighted_indices):
        patch = get_sample(instance_id, index)
        if patch["normalized_patch"] == last_normilized_patch:
            assert len(all_patches) > 0
            all_patches[-1]["repeat"] += 1
            continue
        last_normilized_patch = patch["normalized_patch"]
        cur_patch = patch['patch']
        if last_normilized_patch == "":
            continue
        
        all_patches.append({
            "model_patch": cur_patch,
            "repeat": 1,
            "raw_edit": patch["raw_edit"],
            "prompt": patch["prompt"],
            "raw_model_patch": patch["raw_model_patch"],
            "original_file_content": patch["original_file_content"],
            "belong": patch["belong"],
        })
    
    with open(args.output_file, "a") as f:
        f.write(json.dumps({
            "instance_id": instance_id,
            "all_patches": all_patches
        }) + "\n")

def parse_and_run(args):
    bench = load_dataset(args.dataset, split=args.split)
    existing_instance_ids = load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    
    # separate the patch folders
    output_folders = [Path(folder) for folder in args.patch_folder.split(",")]
    num_folders = len(output_folders)
    # output_folder = Path(args.patch_folder)
    selected_ids = list(range(int(args.num_samples / num_folders)))

    target_ids = list(execution_results.keys())
    instances = [x for x in bench if x["instance_id"] in target_ids]
    
    temp_instances = {
        instance['instance_id'] : {
            "model_name_or_path": 'aglibro',
            "instance_id": instance['instance_id'],
            "model_patch": "<temp>"
        }
        for instance in instances
    }
    dataset = get_dataset_from_preds(args.dataset, args.split, target_ids, temp_instances, "temp")
    test_specs = list(map(make_test_spec, dataset))
    
    # run reproduction in parallel
    print(f"Running {len(target_ids)} instances...")
    with tqdm(total=len(target_ids), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    parse_and_run_instance,
                    [instance for instance in instances if instance['instance_id'] == instance_id][0],
                    [test_spec for test_spec in test_specs if test_spec.instance_id == instance_id][0],
                    args,
                    existing_instance_ids
                ): None
                for instance_id in target_ids
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    
    parser.add_argument("--patch_folder", type=str)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=11)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--plausible", action="store_true")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="success_patches.jsonl")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generating of instance id's which already contain a localization in the output file.",
    )
    
    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)

    # first normalize
    normalize_patches(args)
    # then load results
    _load_results(args)
    # then rerank
    # majority_voting(args)

    parse_and_run(args)

if __name__ == "__main__":
    main()
#
