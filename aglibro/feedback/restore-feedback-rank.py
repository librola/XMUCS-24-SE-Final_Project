import argparse
import concurrent.futures
import copy
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import unified_diff
from pathlib import Path

import docker
from bs4 import BeautifulSoup
from datasets import load_dataset
from docker.models.containers import Container
from markdownify import markdownify as md
from swebench.harness.constants import (APPLY_PATCH_FAIL, APPLY_PATCH_PASS,
                                        KEY_INSTANCE_ID,
                                        MAP_REPO_VERSION_TO_SPECS,
                                        RUN_EVALUATION_LOG_DIR)
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import (EvaluationError,
                                             get_dataset_from_preds)
from swebench.harness.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import load_swebench_dataset, str2bool
from tqdm import tqdm

from aglibro.docker.docker_build import (INSTANCE_IMAGE_BUILD_DIR,
                                           BuildImageError, build_container,
                                           build_env_images, close_logger,
                                           setup_logger)
from aglibro.docker.docker_utils import (clean_images, cleanup_container,
                                           copy_to_container,
                                           exec_run_with_timeout, list_images,
                                           remove_image, should_remove)
from aglibro.feedback.test import make_test_script, run_test
from aglibro.repair.repair import (_post_process_multifile_repair,
                                     construct_topn_file_context,
                                     post_process_raw_output)
from aglibro.libro.llm_prompt import generate_tests
from aglibro.libro.llm_regenerate import regenerate_tests
from aglibro.libro.postprocess import run_generate_test
from aglibro.util.api_requests import num_tokens_from_messages
from aglibro.util.model import get_model_price, make_model
from aglibro.util.postprocess_data import (
    check_code_differ_by_just_empty_lines, check_syntax, extract_python_blocks,
    fake_git_repo, lint_code, normalize_patch, parse_diff_edit_commands,
    parse_edit_commands, remove_empty_lines, split_edit_multifile_commands)
from aglibro.util.postprocess_tests import (  # make_test_script,
    MAP_REPO_TO_TEST_PATH, extract_new_file_content, get_logs_eval,
    get_logs_eval_with_repo, parse_output)
from aglibro.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions, get_repo_structure,
    line_wrap_content, transfer_arb_locs_to_locs)
from aglibro.util.utils import (load_existing_instance_ids,  # setup_logger,
                                  load_json, load_jsonl)
from aglibro.feedback.feedback import (trans_edit_to_patch)

def emulate_iterate(
    instance_id: str,
    instance_dir: Path,
    patch_id: int,
    logger: logging.Logger,
    test: str,
    edit: dict,
    loc: dict,
    file_contents, file_loc_intervals,
    args: argparse.Namespace,
):
    logger.info(f"Iterating instance {instance_id} with patch #{patch_id}")
    
    patch_json_file = instance_dir / f"{instance_id}_{patch_id}.json"
    with open(patch_json_file, "r") as f:
        report = json.load(f)
    assert 'final_status' in report and 'status' in report
    
    logger.info(f"Loaded patch report from {patch_json_file}")
    if report['final_status'] != "PASSED" or report['status'] != 'success':
        assert report['patch'] == report['log'][0]['patch'] == edit['model_patch']
        assert report['final_libro_test_pass_num'] == 0
        logger.info(f"This patch didn't pass libro tests.")
        
        trans_res = {
            "model_patch": edit['model_patch'],
            "raw_model_patch": edit['raw_model_patch'],
            "original_file_content": edit['original_file_content']
        }
        logger.info(f"trans_res = {trans_res}")
        logger.info(f"patch = {repr(report['patch'])}")
        return {
            "id": patch_id,
            "patch": report['log'][0]['patch'],
            "final_libro_test_pass_num": 0,
            "final_status": report['final_status'],
            "final_result": report['final_result'],
            "trans_res": trans_res
        }
    else:
        logger.info(f'This patch passed libro tests.')
        assert report['log'][-1]['type'] == 'test' and report['log'][-1]['result']['success'] == 'PASSED'
        if report['log'][-2]['type'] == 'llm':
            logger.info(f'This patch was modified by LLM.')
            if 'trans_res' in report['log'][-2]:
                logger.info(f'Found `trans_res` in the log, will use it.')
                trans_res = report['log'][-2]['trans_res']
                logger.info(f"trans_res = {trans_res}")
                logger.info(f"patch = {repr(report['patch'])}")
                return {
                    "id": patch_id,
                    "patch": report['patch'],
                    "final_libro_test_pass_num": 1,
                    "final_status": report['final_status'],
                    "final_result": report['final_result'],
                    "trans_res": trans_res
                }
            else:
                logger.info(f"`trans_res` not found in the log, will regenerate it.")
                traj = report['log'][-2]['traj']
                assert not traj['stop']
                trans_res = trans_edit_to_patch(traj['response'], logger, traj, loc, file_contents, file_loc_intervals, args)
                assert report['patch'] == report['log'][-2]['patch'] == trans_res['model_patch']
                logger.info(f"trans_res = {trans_res}")
                logger.info(f"patch = {repr(report['patch'])}")
                return {
                    "id": patch_id,
                    "patch": report['patch'],
                    "final_libro_test_pass_num": 1,
                    "final_status": report['final_status'],
                    "final_result": report['final_result'],
                    "trans_res": trans_res
                }
        else:
            logger.info(f'This patch was not modified by LLM. It originally passed libro tests.')
            trans_res = {
                "model_patch": edit['model_patch'],
                "raw_model_patch": edit['raw_model_patch'],
                "original_file_content": edit['original_file_content']
            }
            logger.info(f"trans_res = {trans_res}")
            logger.info(f"patch = {repr(report['patch'])}")
            return {
                "id": patch_id,
                "patch": report['log'][0]['patch'],
                "final_libro_test_pass_num": 1,
                "final_status": report['final_status'],
                "final_result": report['final_result'],
                "trans_res": trans_res
            }

def emulate_feedback(instance: dict, instance_dir: Path, edits: dict, tests: dict, locs: dict, args: argparse.Namespace):
    try:
        instance_id = instance['instance_id']
        output_dir: Path = results_dir / args.output_dir / 'instance_logs' / instance_id
        
        if args.skip_existing:
            if (output_dir / f"{instance_id}.json").exists():
                return
            elif (output_dir / f"{instance_id}.log").exists():
                # 删除
                (output_dir / f"{instance_id}.log").unlink()
        
        tests = [ x['test'] for x in tests['final_tests']]
        if args.sort_tests:
            tests.sort(key=lambda x: len(x))
        tests = tests[:args.tests_top_n]
        edits = edits['edits'][:args.patches_top_n]
        
        logger = setup_logger(instance_id, output_dir / f"{instance_id}.log")
        logger.info(f"Processing instance {instance_id}")
        
        file_contentss = []
        file_loc_intervalss = []
        for loc in locs:
            pred_files = loc["found_files"][: args.top_n]
            structure = get_repo_structure(
                instance_id, instance["repo"], instance["base_commit"], "playground"
            )
            files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
            # Construct file contents
            file_contents = dict()
            for i, pred_file in enumerate(pred_files):
                content = None

                for file_content in files:
                    if file_content[0] == pred_file:
                        content = "\n".join(file_content[1])
                        file_contents[pred_file] = content
                        break
                assert content is not None, f"{pred_file} file not found"

            # Construct top-n file context
            file_to_edit_locs = dict()
            for i, pred_file in enumerate(pred_files):
                if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
                    file_to_edit_locs[pred_file] = loc["found_edit_locs"][i]

            topn_content, file_loc_intervals = construct_topn_file_context(
                file_to_edit_locs,
                pred_files,
                file_contents,
                structure,
                context_window=args.context_window,
                loc_interval=args.loc_interval,
                fine_grain_loc_only=args.fine_grain_loc_only,
                add_space=args.add_space,
                no_line_number=args.diff_format,
                sticky_scroll=args.sticky_scroll,
            )
            file_contentss.append(file_contents)
            file_loc_intervalss.append(file_loc_intervals)
        
        iter_result ={-1: [], 0: [], 1: []}

        for i, edit in enumerate(edits):
                bl = edit['belong']
                logger.info(f"Starting iteration {i} (patch {i}).")
                for j, test in enumerate(tests):
                    logger.info(f"Starting iteration {i * len(tests) + j} (patch {i}, test {j}).")
                    res = emulate_iterate(
                        instance_id, instance_dir,
                        i * len(tests) + j,
                        logger,
                        test,
                        edit,
                        locs[bl],
                        file_contentss[bl], file_loc_intervalss[bl],
                        args
                    )
                    res['votes'] = edit['votes']
                    iter_result[res['final_libro_test_pass_num']].append(res)
                    logger.info(f"Finished iteration {i * len(tests) + j} (patch {i}, test {j}).")
                    logger.info(f"Result: {res}")
                if len(tests) == 0:
                    res = {
                        "id": i,
                        "patch": edit['model_patch'],
                        "final_libro_test_pass_num": 0,
                        "final_status": "",
                        "final_result": {},
                        "trans_res": {
                            "model_patch": edit['model_patch'],
                            "raw_model_patch": edit['raw_model_patch'],
                            "original_file_content": edit['original_file_content']
                        },
                        "votes": edit['votes']
                    }
                    iter_result[res['final_libro_test_pass_num']].append(res)
                    logger.info(f"No libro tests, preserving the original patch.")
                    logger.info(f"Result: {res}")
        
        
        # 枚举 instance_dir 下所有形如 {instance_id}_{i}.json 的文件
        iterate_files = list(instance_dir.glob(f"{instance_id}_*.json"))
        assert len(iterate_files) == len(edits) * len(tests), f"len(iterate_files) = {len(iterate_files)}, len(edits) = {len(edits)}, len(tests) = {len(tests)}"
        
        top10 = []
        rank_report = {
            1: {
                "patches": [],
                "voted": []
            },
            0: {
                "patches": [],
                "voted": []
            }
        }
        for i in [1, 0]:
            count_norm_patches = {}
            first_appear_idx = {}
            for res in iter_result[i]:
                try:
                    normalized_patch = normalize_patch(
                        instance_id, res['patch'], res['trans_res']['original_file_content']
                    ).strip()
                except Exception as e:
                    print(f"Error in normalize_patch: ({instance_id}):\n {e}\n {res['patch']}\n")
                    logger.error(f"Error in normalize_patch: ({instance_id}):\n {e}\n {res['patch']}\n")
                    normalized_patch = ""
                res['normalized_patch'] = normalized_patch
                if not normalized_patch:
                    normalized_patch = res['patch'].strip()
                count_norm_patches[normalized_patch] = count_norm_patches.get(normalized_patch, 0) + res['votes']
                if normalized_patch not in first_appear_idx:
                    first_appear_idx[normalized_patch] = res['id']
                
                res['current_total_votes'] = count_norm_patches[normalized_patch]
                res['original_votes'] = res.pop('votes')
                res['original_patch_id'] = res['id'] // len(tests) if len(tests) else res['id']
                res['test_id'] = res['id'] % len(tests) if len(tests) else 0
                logger.info(f"Patch #{res['id']}(patch #{res['original_patch_id']}, test #{res['test_id']}), {'can' if i else 'cannot'} pass libro test: \npatch = {repr(res['patch'])}, \nnormalized_patch = {repr(normalized_patch)}, \nnow has {count_norm_patches[normalized_patch]} votes.")
                rank_report[i]['patches'].append(res.copy())
            count_norm_patches[''] = 0
            first_appear_idx[''] = 999
                
            iter_result[i].sort(key=lambda x: (
                count_norm_patches[x['normalized_patch']],
                -first_appear_idx[x['normalized_patch']],
            ), reverse=True)
            
            last_normilized_patch = None
            for res in iter_result[i]:
                if res['normalized_patch'] == last_normilized_patch:
                    continue
                last_normilized_patch = res['normalized_patch']
                top10.append({
                    "patch": res['patch'],
                    "patch_type": i,
                    "votes": count_norm_patches[res['normalized_patch']],
                })
                rank_report[i]['voted'].append({
                    "id": res['id'],
                    "patch": res['patch'],
                    "normalized_patch": res['normalized_patch'],
                    "votes": count_norm_patches[res['normalized_patch']],
                })
                logger.info(f"Top {len(top10)}: \nOriginal patch #{res['id']}(patch #{res['original_patch_id']}, test #{res['test_id']}), {'can' if i else 'cannot'} pass libro test: \npatch = {repr(res['patch'])}, \nnormalized_patch = {repr(res['normalized_patch'])}, \nhas {count_norm_patches[res['normalized_patch']]} votes.")
            #     if len(top10) >= 10:
            #         break
            # if len(top10) >= 10:
                # break
        
        rank_report['all_sorted'] = top10
        top10 = top10[:10]
        while len(top10) < 10:
            top10.append({
                "patch": "",
                "patch_type": -1,
                "votes": 0
            })
        
        with open(output_dir / f"{instance_id}.json", "w") as f:
            json.dump(rank_report, f, indent=4)
        
        global all_preds
        for i, res in enumerate(top10):
            assert res['patch_type'] == all_preds[instance_id][i]['patch_type'], f"Instance `{instance_id}`: top {i}-th patch type mismatch, should be {all_preds[instance_id][i]['patch_type']}, but calculated {res['patch_type']}."
            assert res['patch'] == all_preds[instance_id][i]['model_patch'], f"Instance `{instance_id}`: top {i}-th patch content mismatch"
            assert res['votes'] == all_preds[instance_id][i]['votes'], f"Instance `{instance_id}`: top {i}-th votes mismatch, should be {all_preds[instance_id][i]['votes']}, but calculated {res['votes']}."
    except Exception as e:
        logger.error(f"Error in emulate_feedback: {e}")
        with open("errorlog.txt", "a") as f:
            f.write(f"Error in emulate_feedback: {e}\n")
        raise

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results/aglibro-feedback-1105")
parser.add_argument("--feedback_dir", type=str, default="feedback")
parser.add_argument("--preds_file", type=str, default="all_preds.jsonl")
parser.add_argument("--output_dir", type=str, default="restore-rank")

parser.add_argument("--loc_files", type=str, required=True)
parser.add_argument("--tests_file", type=str)
parser.add_argument("--edits_file", type=str, required=True)
parser.add_argument("--top_n", type=int, default=1)
parser.add_argument("--loc_interval", action="store_true")
parser.add_argument("--context_window", type=int, default=10)
parser.add_argument("--add_space", action="store_true")
parser.add_argument("--cot", action="store_true")
parser.add_argument("--fine_grain_loc_only", action="store_true")
parser.add_argument("--diff_format", action="store_true")
parser.add_argument("--skip_greedy", action="store_true")
parser.add_argument("--sticky_scroll", action="store_true")
parser.add_argument("--tests_type", type=str, default="libro", choices=["libro", "oracle", "official"])
parser.add_argument("--tests_top_n", type=int, default=2)
parser.add_argument("--sort_tests", action="store_true")
parser.add_argument("--patches_top_n", type=int, default=10)
parser.add_argument("--run_regression", action="store_true")
parser.add_argument("--run_id", type=str, default="temp")
parser.add_argument("--skip_existing", action="store_true")
args = parser.parse_args()

results_dir = Path(args.results_dir)
feedback_dir = results_dir / args.feedback_dir

# 导入最后的结果，以便比对
all_preds = {}
for i in range(10):
    preds_file = feedback_dir / (args.preds_file.replace(".jsonl", f"_{i}.jsonl") if i else args.preds_file)
    preds_i = load_jsonl(preds_file)
    for pred in preds_i:
        all_preds.setdefault(pred[KEY_INSTANCE_ID], []).append(pred)
target_ids = list(all_preds.keys())

locs = dict()
for loc_file in args.loc_files.split(','):
    loc = load_jsonl(loc_file)
    for instance in loc:
        locs.setdefault(instance['instance_id'], []).append(instance)
all_edits = load_jsonl(args.edits_file)
all_tests = load_jsonl(args.tests_file)
all_tests_ids = {edit['instance_id'] for edit in all_tests}
for id in target_ids:
    if id not in all_tests_ids:
        all_tests.append({
            "instance_id": id,
            "final_tests": []
        })

all_edits = {edit['instance_id']: edit for edit in all_edits}
all_tests = {test['instance_id']: test for test in all_tests}

for instance_id, preds in all_preds.items():
    assert len(preds) == 10

bench = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
bench = {instance['instance_id']: instance for instance in bench}


# run in parallel
print(f"Running {len(target_ids)} instances...")
with tqdm(total=len(target_ids), smoothing=0) as pbar:
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Create a future for running each instance
        futures = {
            executor.submit(emulate_feedback, bench[instance_id], feedback_dir / 'instance_logs' / instance_id,
                    all_edits[instance_id], all_tests[instance_id], locs[instance_id], args
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