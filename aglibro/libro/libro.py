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
    make_test_script,
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

def get_instance_image_key(instance):
    return f"sweb.eval.x86_64.{instance['instance_id']}:latest"

def update_test_per_turn(
    test_id: int,
    turn_id: int,
    messages: list[dict],
    traj: list[dict],
    container: docker.models.containers.Container,
    logger: logging.Logger,
    test_spec: TestSpec,
    instance: dict,
    model: str,
    check_temperature: float,
    check_template: str,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    test_command: str
):
    assert traj[-1]["role"] == "assistant"
    cur_test = traj[-1]["test"]
    resolved, output = run_generate_test(
        test_id,
        turn_id,
        container,
        test_spec,
        instance,
        cur_test,
        logger,
        cache_level,
        clean,
        force_rebuild,
        test_command
    )
    traj.append({
        "role": "run",
        "output": output,
        "resolved": resolved,
    })
    
    regenerate_tests(
        test_id,
        turn_id,
        instance,
        logger,
        output,
        resolved,
        messages,
        traj,
        model,
        check_temperature,
        check_template,
    )
    if traj[-1]['success']:
        traj[-1]['test'] = cur_test
        traj[-1]['resolved'] = resolved
    
    return traj[-1]["success"]

def reproduce_instance(
    test_spec: TestSpec,
    instance: dict,
    output_folder: str,
    output_file: str,
    model: str,
    num_samples: int,
    generate_temperature: float,
    check_temperature: float,
    skip_existing: bool,
    generate_template: str,
    check_template: str,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_turns: int,
    cost_limit: float,
    docker_client: docker.DockerClient,
    existing_instance_ids: list[str],
    run_id: str,
):
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    version = instance["version"]
    
    instance_dir = Path(output_folder) / "instance_logs" / str(instance_id)
    instance["instance_dir"] = instance_dir
    log_file = instance_dir / f"{instance_id}.log"
    logger = setup_logger(instance_id, log_file, mode="a")
    logger.info(f"Processing instance {instance_id}")
    
    if skip_existing and existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Instance {instance_id} already exists in {output_file}, skipping.")
        return
    
    prev_intermidiate_tests = None
    if skip_existing and (instance_dir / "intermidiate_tests.json").exists():
        prev_intermidiate_tests = load_jsonl(
            instance_dir / "intermidiate_tests.json"
        )
    _, trajs = generate_tests(
        instance,
        logger,
        instance_dir / "intermidiate_tests.json",
        model,
        num_samples,
        generate_temperature,
        skip_existing,
        generate_template,
        prev_intermidiate_tests,
    )
    
    cost_per_input_token, cost_per_output_token = get_model_price(model)
    def calc_cost(usage):
        return (
            usage["prompt_tokens"] * cost_per_input_token +
            usage["completion_tokens"] * cost_per_output_token
        )
        
    # Build + start instance container (instance image should already be built)
    container = build_container(test_spec, docker_client, run_id, logger, False, False)
    container.start()
    logger.info(f"Container for {instance_id} started: {container.id}")
    
    
    try:
        # Copy test script to container
        test_script = instance_dir / "do_test.sh"
        spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
        test_script_content, test_command = make_test_script(instance, spec, "testbed", "/testbed", instance["base_commit"], "", True)
        test_script.write_text(test_script_content)
        copy_to_container(container, test_script, Path("/do_test.sh"))
        logger.info(f"Test script for {instance_id} written to {test_script}, copied to container as /do_test.sh")
        
        final_tests = []
        instance_prompt_tokens, instance_completion_tokens, instance_total_price = 0, 0, 0.0
        for i, _traj in enumerate(trajs):
            if _traj["gen_test"].strip() == "":
                continue
            
            messages = [
                *_traj["prompt"],
                {
                    "role": "assistant",
                    "content": _traj["response"],
                }
            ]
            traj = copy.deepcopy(messages)
            prompt_tokens, completion_tokens, total_price = 0, 0, 0.0
            def update_tokens(usage):
                nonlocal prompt_tokens, completion_tokens, total_price
                usage["cost"] = calc_cost(usage)
                prompt_tokens += usage["prompt_tokens"]
                completion_tokens += usage["completion_tokens"]
                total_price += usage["cost"]
            
            update_tokens(_traj["usage"])
            for tr in traj:
                if tr["role"] == "assistant":
                    tr["usage"] = _traj["usage"]
                    tr["test"] = _traj["gen_test"]
                    break
            
            def write_current_status_to_file(turn, status):
                nonlocal messages, traj, prompt_tokens, completion_tokens, total_price
                with open(instance_dir / f"{instance_id}_test_{i}.json", "w") as f:
                    json.dump(
                        {
                            "messages": messages,
                            "traj": traj,
                            "result": "running" if status is None else status,
                            "turns": turn + 1,
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "cost": total_price
                            }
                        }, f, indent=4
                    )
                
            final_status = None
            for turn in range(max_turns):
                write_current_status_to_file(turn, final_status)
                
                success = update_test_per_turn(
                    i,
                    turn,
                    messages,
                    traj,
                    container,
                    logger,
                    test_spec,
                    instance,
                    model,
                    check_temperature,
                    check_template,
                    cache_level,
                    clean,
                    force_rebuild,
                    test_command
                )
                update_tokens(traj[-1]["usage"])
                if success:
                    final_status = "success"
                    logger.info(f"{instance_id} test #{i} successfully reproduced at turn {turn}.")
                    break
                else:
                    logger.info(f"{instance_id} test #{i} doesn't reproduce at turn {turn}, continue.")
                if total_price > cost_limit:
                    final_status = "exit_cost"
                    logger.info(f"Cost limit reached for {instance_id} test #{i}.")
                    break
                if turn == max_turns - 1:
                    final_status = "exit_turn"
                    logger.info(f"Max turns reached for {instance_id} test #{i}.")
            
            assert final_status is not None
            write_current_status_to_file(turn, final_status)
            if final_status == "success":
                final_tests.append({
                    "test": traj[-1]['test'],
                    "resolved_before": traj[-1]['resolved'],
                })
            instance_prompt_tokens += prompt_tokens
            instance_completion_tokens += completion_tokens
            instance_total_price += total_price
            with open(instance_dir / f"{instance_id}_final_tests.json", "w") as f:
                json.dump({
                    "final_tests": final_tests,
                    "usage": {
                        "prompt_tokens": instance_prompt_tokens,
                        "completion_tokens": instance_completion_tokens,
                        "cost": instance_total_price
                    }
                }, f, indent=4)
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
        return
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
        return
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                    f"{traceback.format_exc()}\n"
                    f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
        print(error_msg)
        return
    finally:
        # Remove instance container + image, close logger
        cleanup_container(docker_client, container, logger)
        
    # 按长度排序
    final_tests.sort(key=lambda x: len(x['test']))
    
    with open(output_file, "a") as f:
        f.write(json.dumps({
            "instance_id": instance_id,
            "final_tests": final_tests
        }) + "\n")
        logger.info(f"Final tests for {instance_id} written to {output_file}")
    
def reproduce(
    dataset: str,
    split: str,
    output_folder: str,
    output_file: str,
    model: str,
    num_samples: int,
    generate_temperature: float,
    check_temperature: float,
    num_threads: int,
    target_ids: list[str],
    skip_existing: bool,
    generate_template: str,
    check_template: str,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_turns: int,
    cost_limit: float,
    run_id: str
):
    bench = load_dataset(dataset, split=split)
    if target_ids is None:
        target_ids = [x["instance_id"] for x in bench]
    instances = [x for x in bench if x["instance_id"] in target_ids]
    existing_instance_ids = load_existing_instance_ids(output_file) if skip_existing else set()
    
    # get test_specs which required by SWE-Bench
    temp_instances = {
        instance['instance_id'] : {
            "model_name_or_path": model,
            "instance_id": instance['instance_id'],
            "model_patch": "<temp>"
        }
        for instance in instances
    }
    dataset = get_dataset_from_preds(dataset, split, target_ids, temp_instances, run_id)
    test_specs = list(map(make_test_spec, dataset))
    
    # start docker and build environment images
    client = docker.from_env()
    print("Docker client connected.")
    build_env_images(client, dataset, False, num_threads)
    print("Environment images built.")
    
    # print number of existing instance images
    instance_image_ids = {get_instance_image_key(instance) for instance in instances}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")
    
    # run reproduction in parallel
    print(f"Running {len(target_ids)} instances...")
    with tqdm(total=len(target_ids), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    reproduce_instance,
                    test_spec,
                    [instance for instance in instances if instance["instance_id"] == test_spec.instance_id][0],
                    output_folder,
                    output_file,
                    model,
                    num_samples,
                    generate_temperature,
                    check_temperature,
                    skip_existing,
                    generate_template,
                    check_template,
                    cache_level,
                    clean,
                    force_rebuild,
                    max_turns,
                    cost_limit,
                    client,
                    existing_instance_ids,
                    run_id
                ): None
                for test_spec in test_specs
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
    
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="reproduce_tests.jsonl")
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        # choices=["gpt-4o", "deepseek-coder", "gpt-4o-mini"],
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of tests to generate initially per instance")
    parser.add_argument("--generate_temperature", type=float, default=0.0)
    parser.add_argument("--check_temperature", type=float, default=0.0)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    
    parser.add_argument("--target_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generating of instance id's which already contain a localization in the output file.",
    )
    
    parser.add_argument('--generate_template', default='2example_chat')
    parser.add_argument('--check_template', default='regenerate_prompt_singal')
    
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="instance",
    )
    parser.add_argument("--clean", type=str2bool, default=False, help="Clean images above cache level")
    parser.add_argument('--force_rebuild', action='store_true', default=False, help="Force rebuild of instance images")
    
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--cost_limit", type=float, default=0.2)
    parser.add_argument("--run_id", type=str, default="temp")
    
    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    
    reproduce(
        dataset=args.dataset,
        split=args.split,
        output_folder=args.output_folder,
        output_file=args.output_file,
        model=args.model,
        num_samples=args.num_samples,
        generate_temperature=args.generate_temperature,
        check_temperature=args.check_temperature,
        num_threads=args.num_threads,
        target_ids=args.target_ids,
        skip_existing=args.skip_existing,
        generate_template=args.generate_template,
        check_template=args.check_template,
        cache_level=args.cache_level,
        clean=args.clean,
        force_rebuild=args.force_rebuild,
        max_turns=args.max_turns,
        cost_limit=args.cost_limit,
        run_id=args.run_id,
    )

if __name__ == "__main__":
    main()