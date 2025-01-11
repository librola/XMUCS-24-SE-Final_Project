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
    parse_error_info
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

TEMPLATE_DIR = 'aglibro/libro/prompt_templates/'

def get_instance_image_key(instance):
    return f"sweb.eval.x86_64.{instance['instance_id']}:latest"

def load_souce_tests(souce_tests_dir: Path, instance_id: str):
    res = []

    target_dir = souce_tests_dir / instance_id
    
    pattern = re.compile(rf'{instance_id}_test_\d+\.json')
    
    if target_dir.exists() and target_dir.is_dir():
        for file in target_dir.iterdir():
            if file.is_file() and pattern.match(file.name):
                
                with open(file, "r") as f:
                    test_traj = json.load(f)
                if test_traj['result'] == 'success':
                    result = (test_traj['traj'][-1]['test'], test_traj['traj'][-4]['output'])
                    res.append(result)
    
    return res

def make_messages_from_file(problem_statement, repo, version, test_groups,
                          template_file=TEMPLATE_DIR+'select_prompt.json'):

    with open(template_file) as f:
        messages = json.load(f)

        for msg in messages:
            example_text_path = re.findall(r'{%(.+?)%}', msg['content'])
            if len(example_text_path) > 0:
                for ef in example_text_path:
                    with open(os.path.join(TEMPLATE_DIR, ef)) as f:
                        example_text = f.read()
                    msg['content'] = msg['content'].replace('{%'+ef+'%}', example_text)
        
        output_choices = ""
        for i, test_group in enumerate(test_groups):
            output_choices += chr(i + ord('A')) + '\n'
            output_choices += "```\n"
            output_choices += test_group['output']
            output_choices += "```\n\n"
            
        current_query = messages[-1]['content']
        current_query = current_query.replace('{{bug_report_content}}', problem_statement) \
                                     .replace('{{repo}}', repo) \
                                     .replace('{{version}}', version) \
                                     .replace('{{output_choices}}', output_choices)
        messages[-1]['content'] = current_query

    return messages

def ask_llm_to_select(
    instance: dict,
    test_groups: list[dict],
    model,
    temperature: float,
    template: str,
    logger,
    record
):
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    version = instance["version"]
    problem_statement = instance["problem_statement"]
    prompt = make_messages_from_file(
        problem_statement,
        repo,
        version,
        test_groups,
        template_file=TEMPLATE_DIR + template + ".json")

    logger.info(f"Prompt: {prompt}")
    
    maked_model = make_model(
        model = model,
        logger = logger,
        backend = 'openai' if model.startswith('gpt') else 'deepseek',
        temperature = temperature,
        max_tokens = 1024,
        batch_size = 1
    )
    trajs = maked_model.codegen(prompt, num_samples=1)
    assert len(trajs) == 1, f"Expected 1 trajectory, got {len(trajs)}"
    
    traj = trajs[0]
    query_result = traj["response"].strip()
    choice = query_result.split('\n')[-1].strip()
    if choice in ['A', 'B', 'C', 'D'] and ord(choice) - ord('A') < len(test_groups):
        choice = ord(choice) - ord('A')
    else:
        choice = 0
    
    usage = traj["usage"]
    cost_per_input_token, cost_per_output_token = get_model_price(model)
    usage['cost'] = cost_per_input_token * usage['prompt_tokens'] + cost_per_output_token * usage['completion_tokens']
    
    record['traj'].append({
        "tests": [(test_group["test"], test_group["parsed_output"]) for test_group in test_groups],
        "response": traj["response"],
        "choice": choice,
        "usage": traj["usage"],
    })
    
    record["usage"]["prompt_tokens"] += usage["prompt_tokens"]
    record["usage"]["completion_tokens"] += usage["completion_tokens"]
    record["usage"]["cost"] += usage["cost"]
    
    return choice


def ask_llm_to_sort(
    instance: dict,
    test_groups: list[dict],
    model,
    temperature: float,
    template: str,
    logger,
    record
):
    if len(test_groups) == 0:
        return []
    if len(test_groups) == 1:
        return [0]
    
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    version = instance["version"]
    problem_statement = instance["problem_statement"]
    prompt = make_messages_from_file(
        problem_statement,
        repo,
        version,
        test_groups,
        template_file=TEMPLATE_DIR + "sort_prompt" + ".json")

    logger.info(f"Prompt: {prompt}")
    
    maked_model = make_model(
        model = model,
        logger = logger,
        backend = 'openai' if model.startswith('gpt') else 'deepseek',
        temperature = temperature,
        max_tokens = 1024,
        batch_size = 1
    )
    trajs = maked_model.codegen(prompt, num_samples=1)
    assert len(trajs) == 1, f"Expected 1 trajectory, got {len(trajs)}"
    
    traj = trajs[0]
    query_result = traj["response"]
    if ("```") in query_result:
        query_result = query_result.split("```")[1]
    else:
        query_result = query_result
    query_result = query_result.strip()
    
    choices = query_result.split('\n')[-1].strip()
    
    import string
    alphabet = string.ascii_uppercase

    valid_choices = []
    seen_letters = set()
    for char in choices:
        char = char.upper()
        if char in alphabet:
            if char not in seen_letters:
                seen_letters.add(char)
                valid_choices.append(char)

    result = []

    seen_indices = set()
    for char in valid_choices:
        index = alphabet.index(char)
        if index < len(test_groups):
            result.append(index)
            seen_indices.add(index)

    for i, group in enumerate(test_groups):
        if i not in seen_indices:
            result.append(i)
    
    usage = traj["usage"]
    cost_per_input_token, cost_per_output_token = get_model_price(model)
    usage['cost'] = cost_per_input_token * usage['prompt_tokens'] + cost_per_output_token * usage['completion_tokens']
    
    record['traj'].append({
        "tests": [(test_group["test"], test_group["parsed_output"]) for test_group in test_groups],
        "response": traj["response"],
        "choices": choices,
        "result": result,
        "usage": traj["usage"],
    })
    
    record["usage"]["prompt_tokens"] += usage["prompt_tokens"]
    record["usage"]["completion_tokens"] += usage["completion_tokens"]
    record["usage"]["cost"] += usage["cost"]
    
    return result

def group_instance(
    instance: dict,
    source_tests: dict | Path,
    output_folder: str,
    output_file: str,
    model: str,
    temperature: float,
    template: str,
    skip_existing: bool,
    existing_instance_ids: list[str],
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
    
    if type(source_tests) == dict:
        source_tests = source_tests['final_tests']
    else:
        logger.info(f"Loading source tests from {source_tests}")
        source_tests = load_souce_tests(source_tests, instance_id)
    
    map_output_to_tests = {}
    logger.info(f'Parsing {len(source_tests)} tests...')
    for test, output in source_tests:
        # parsed_error_info = parse_error_info(output, repo, version)
        parsed_error_info = output
        map_output_to_tests[parsed_error_info] = \
            map_output_to_tests.get(parsed_error_info, []) \
            + [{
                "test": test,
                "output": output,
                "parsed_output": parsed_error_info
            }]
        logger.info(f'One test parsed: {parsed_error_info}')
    
    map_output_to_test = {}
    for parsed_output, tests in map_output_to_tests.items():
        tests.sort(key=lambda x: len(x['test']))
        map_output_to_test[parsed_output] = tests[0]
        
    logger.info(f'Parsed. Divided into {len(map_output_to_tests)} groups.')

    record = {
        "traj": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
        }
    }
    
    logger.info(f"Selecting tests for {instance_id} out of {len(map_output_to_tests)} groups...")

    map_output_to_test_items = list(map_output_to_test.items())
    # while len(map_output_to_test_items) > 1:
    next_round_items = []

    for i in range(0, len(map_output_to_test_items), 4):
        group = map_output_to_test_items[i:i + 4]
        selected_group = [ x[1] for x in group ]
        selected_index = ask_llm_to_sort(instance, selected_group, model, temperature, template, logger, record)
        if len(selected_index) > 2:
            selected_index = selected_index[:2]
        else:
            selected_index = selected_index[:1]
        next_round_items.extend([group[x] for x in selected_index])
        
        with open(instance_dir / f"{instance_id}.json", "w") as f:
            json.dump(record, f, indent=4)

    map_output_to_test_items = next_round_items
    group = map_output_to_test_items
    selected_group = [ x[1] for x in group ]
    selected_index = ask_llm_to_sort(instance, selected_group, model, temperature, template, logger, record)[:3]
    final_tests = [group[x] for x in selected_index]
    
    record['final_tests'] = [x[1] for x in final_tests]
    # if len(map_output_to_test_items) == 0:
    #     record["final_test"] = None
    # else:
    #     record["final_test"] = map_output_to_test_items[0][1]
    with open(instance_dir / f"{instance_id}.json", "w") as f:
        json.dump(record, f, indent=4)
    
    with open(output_file, "a") as f:
        f.write(json.dumps({
            "instance_id": instance_id,
            "final_tests": [{
                "test": test['test'],
                "resolved_before": False
            } for test in record['final_tests']],
        }) + "\n")

def group_tests(
    dataset: str,
    split: str,
    output_folder: str,
    output_file: str,
    source_tests_file: str,
    model: str,
    temperature: float,
    template: str,
    num_threads: int,
    target_ids: list[str],
    skip_existing: bool,
):
    get_model_price(model)
    dataset = load_dataset(dataset, split=split)
    if Path(source_tests_file).is_file():
        source_tests = load_jsonl(source_tests_file)
        source_tests = { x['instance_id'] : x for x in source_tests }
    else:
        source_tests = None
        source_tests_file = Path(source_tests_file)
    existing_instance_ids = load_existing_instance_ids(output_file) if skip_existing else set()
    
    target_instances = [instance for instance in dataset if instance["instance_id"] in target_ids]
    
    print(f"Running {len(target_ids)} instances...")
    with tqdm(total=len(target_ids), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    group_instance,
                    instance,
                    source_tests[instance['instance_id']] if source_tests else source_tests_file,
                    output_folder,
                    output_file,
                    model,
                    temperature,
                    template,
                    skip_existing,
                    existing_instance_ids,
                ): None
                for instance in target_instances
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
    parser.add_argument("--source_tests_file", type=str, required=True)
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        # choices=["gpt-4o", "deepseek-coder", "gpt-4o-mini"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument('--template', default='select_prompt')
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
    
    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    
    group_tests(
        args.dataset,
        args.split,
        args.output_folder,
        args.output_file,
        args.source_tests_file,
        args.model,
        args.temperature,
        args.template,
        args.num_threads,
        args.target_ids,
        args.skip_existing,
    )

if __name__ == "__main__":
    main()