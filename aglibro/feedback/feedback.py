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
    extract_new_file_content
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

repair_relevant_file_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
"""
repair_relevant_file_with_scope_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
In the file below, "..." refers to some less relevant content being omited for brebity.
"""
with_scope_explanation = """
Note that "..." refers to some omited content that is not actually in the files. Your *SEARCH/REPLACE* edit must not contain such "...".
"""
repair_relevant_file_with_suspicious_loc_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs. Some suspicious locations are provided for closer inspection.
"""

repair_prompt_regression_failed = """
{pre_template}
In ordered to fix the issue, we have first localized the bug based on the issue statement, and then generated *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.

Here is one generated *SEARCH/REPLACE* edit which we once expected to fix the issue:
--- BEGIN GENERATED EDIT ---
```python
{generated_edit}
```
--- END GENERATED EDIT ---

We apply the *SEARCH/REPLACE* edit to the codebase and then run the tests to see if the it could pass the regression tests. The test command is `{test_command}`. But it failed, the output is as follows:
--- BEGIN OUTPUT ---
{output}
--- END OUTPUT ---

You should generate a new *SEARCH/REPLACE* edit to fix the issue. NOTE that the *SEARCH/REPLACE* edit should be based on the original code, not the code after the previous *SEARCH/REPLACE* edit.
"""

repair_prompt_libro_failed = """
{pre_template}
In ordered to fix the issue, we have first localized the bug based on the issue statement, and then generated *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.

Here is one generated *SEARCH/REPLACE* edit which we once expected to fix the issue:
--- BEGIN GENERATED EDIT ---
```python
{generated_edit}
```
--- END GENERATED EDIT ---

Besides, we construct one test case which is expected to reproduce the issue (note that the test case is not guaranteed to reproduce the issue).
We apply the *SEARCH/REPLACE* edit to the codebase and then run the test above to check if the issue is fixed. The test command is `{test_command}`. However, the test failed, so we present the test and the output below:
{test_output_pairs}

If you think the *SEARCH/REPLACE* edit is incorrect, you should generate a new *SEARCH/REPLACE* edit to fix the issue. NOTE that the *SEARCH/REPLACE* edit should be based on the original code, not the code after the previous *SEARCH/REPLACE* edit.
If you think the test case cannot reproduce the issue or the test cases are incorrect, you should do some analysis and then reply with a singal line which contains the word 'FINISH' without any other word or character at the end of your response.

NOTE:
1. If you want to modify the *SEARCH/REPLACE* edit, please wrap the *SEARCH/REPLACE* edit in blocks ```python...```, and make sure that your response should exactly contain one the ```python...``` block!
2. Please do not try to modify the test case. If you think the test cases are incorrect, just tell your thought and then reply 'FINISH'. It is not your responsibility and right to modify the test cases.
3. Please clearly indicate the file path and the code content you want to modify in the *SEARCH/REPLACE* edit. The indent of the code content should be consistent with the original code. DO NOT remove any space or add any space in the code content, and how many spaces are there in the original code, how many spaces should be in the code content.
"""

repair_prompt_edit_wrong = """
{pre_template}
In ordered to fix the issue, we have first localized the bug based on the issue statement, and then generated *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.

Here is one generated *SEARCH/REPLACE* edit.
--- BEGIN GENERATED EDIT ---
```python
{generated_edit}
```
--- END GENERATED EDIT ---

However, the *SEARCH/REPLACE* edit is incorrect. We cannot search the code content in the original code. Please refer to the generated *SEARCH/REPLACE* edit and the file content we provide to generate a new *SEARCH/REPLACE* edit to fix the issue.
The error of the *SEARCH/REPLACE* edit may be caused by the following reasons:
1. The *SEARCH/REPLACE* edit is not based on the original code.
2. The spaces of the *SEARCH/REPLACE* edit are not consistent with the original code, especially the indent.
3. The file name of the *SEARCH/REPLACE* edit is incorrect.

Please generate a new *SEARCH/REPLACE* edit. Please wrap the *SEARCH/REPLACE* edit in your answer in blocks ```python...```, and make sure that your response should exactly contain one the ```python...``` block!
"""

test_output_pair_template = """
--- BEGIN TEST INFO ---
--- BEGIN TEST CONTENT ---
```python
{test_content}
```
--- END TEST CONTENT ---

--- BEGIN OUTPUT ---
```
{output}
```
--- END OUTPUT ---
--- END TEST INFO ---
"""

test_output_pair_template_oracle = """
--- BEGIN TEST INFO ---
Note that the test cases are represented as a patch (like a git diff).
--- BEGIN TEST CONTENT ---
{test_patch}
--- END TEST CONTENT ---

{outputs}
--- END TEST INFO ---
"""

test_output_pair_template_oracle_output = """
--- BEGIN OUTPUT of CASE `{case_name}` ---
```
{output}
```
--- END OUTPUT of CASE `{case_name}` ---
"""

def get_instance_image_key(instance):
    return f"sweb.eval.x86_64.{instance['instance_id']}:latest"

def make_messages(fail_type, initial_prompt, edit, test_result):
    
    initial_prompt = initial_prompt.split('Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.')[0]
    assert initial_prompt.endswith('\n\n')
    
    
    if fail_type == 'REGRESSION':
        message = repair_prompt_regression_failed
        message = message.format(
            pre_template=initial_prompt,
            generated_edit=edit,
            test_command=test_result['test_command'],
            output=test_result['output']
        )
    elif fail_type == 'LIBRO':
        message = repair_prompt_libro_failed
        message = message.format(
            pre_template=initial_prompt,
            generated_edit=edit,
            test_command=test_result['res'][0]['test_command'],
            test_output_pairs='\n'.join(
                test_output_pair_template.format(
                    # id=i+1,
                    test_content=test['test'],
                    output=test['output']
                )
                for i, test in enumerate(test_result['res'])
                if test['success'] == 'LIBRO'
            ),
        )
    elif fail_type == 'EDIT_WRONG':
        message = repair_prompt_edit_wrong
        message = message.format(
            pre_template=initial_prompt,
            generated_edit=edit
        )
    elif fail_type == "ORACLE":
        message = repair_prompt_libro_failed
        message = message.format(
            pre_template=initial_prompt,
            generated_edit=edit,
            test_command=test_result['test_command'],
            test_output_pairs=test_output_pair_template.format(
                test_content=test_result['test_patch'],
                output=test_result['output']
            )
            # test_output_pairs=test_output_pair_template_oracle.format(
            #     test_patch=test_result['test_patch'],
            #     outputs='\n\n'.join(
            #         test_output_pair_template_oracle_output.format(
            #             case_name=test,
            #             output=test_result['error_info'][test]
            #         )
            #         for test, res in test_result['report'].items()
            #         if res == 'ERROR' or res == 'FAILED'
            #     )
            # )
        )
        
    messages = [
        {
            "role": "user",
            "content": message,
        }
    ]
    
    return messages

def query_llm(initial_prompt, edit, test_result, model, temperature, logger):
    
    prompt = make_messages(test_result['success'], initial_prompt, edit, test_result)
    logger.info(f"Prompt: {prompt[0]['content']}")
    # exit(0)
    
    model_ = make_model(
        model = model,
        logger = logger,
        backend = 'openai' if model.startswith('gpt') else ('deepseek' if model.startswith('deepseek') else 'claude'),
        temperature = temperature,
        max_tokens = 1024,
        batch_size = 1
    )
    trajs = model_.codegen(prompt, num_samples=1)
    assert len(trajs) == 1, f"Expected 1 trajectory, got {len(trajs)}"
    
    traj = trajs[0]
    query_result = traj["response"].strip()
    
    if '```python\n' in query_result:
        query_result = query_result.split('```python\n')[1].split('```')[0]
        stop = False
    else:
        query_result = ""
        stop = True
        
    cost_per_input_token, cost_per_output_token = get_model_price(model)
    def calc_cost(usage):
        return (
            usage["prompt_tokens"] * cost_per_input_token +
            usage["completion_tokens"] * cost_per_output_token
        )
    
    traj['usage']['cost'] = calc_cost(traj['usage'])
    traj['prompt'] = prompt[0]['content']
    traj['result'] = query_result
    traj['stop'] = stop
    
    return traj

def trans_edit_to_patch(raw_output, logger, traj, loc, file_contents, file_loc_intervals, args):
    logger.info(f"raw output:\n{raw_output}")
    
    if raw_output == "":
        return ""
    
    all_generations = []
    all_generations.append(raw_output)
    
    edited_file, new_content = _post_process_multifile_repair(
        raw_output,
        file_contents,
        logger,
        file_loc_intervals,
        diff_format=args.diff_format,
    )

    if new_content == "":
        prev_content = ""
        file_name = ""
    else:
        prev_content = file_contents[edited_file]
        prev_content = prev_content
        file_name = edited_file
        
    raw_output = {
        # "instance_id": instance_id,
        "raw_output": raw_output,
        "all_generations": raw_output,
        "try_count": 0,
        "traj": traj,
        "prev_content": prev_content,
        "file_names": file_name,
    }
    
    assert raw_output["raw_output"] != ""
    
    try:
        raw_output_text = raw_output["all_generations"]
        original_file_content = raw_output["prev_content"]
        pred_file = raw_output["file_names"]
        
        pred_files = loc["found_files"][: args.top_n]
        
        git_diffs = ""
        raw_git_diffs = ""
        if isinstance(raw_output["raw_output"], str):
            # for backward compatibility
            raw_output["raw_output"] = [raw_output["raw_output"]]

        file_contents = {pred_file: original_file_content}

        file_loc_intervals = dict()
        
        for i, tmp_pred_file in enumerate(pred_files):
            if tmp_pred_file != pred_file:
                continue
            if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
                line_locs, context_intervals = transfer_arb_locs_to_locs(
                    loc["found_edit_locs"][i],
                    None,
                    loc["found_files"][i],
                    args.context_window,
                    args.loc_interval,
                    args.fine_grain_loc_only,
                    file_content=file_contents[pred_file]
                    if pred_file in file_contents
                    else "",
                )
            else:
                line_locs, context_intervals = [], []  # default values.

            file_loc_intervals[pred_file] = context_intervals
    except Exception as e:
        logger.info(e)
        print(e)
        raw_output_text = ""
    
    if raw_output_text:
        git_diffs, raw_git_diffs, content = post_process_raw_output(
            raw_output_text, file_contents, logger, file_loc_intervals, args
        )
    else:
        git_diffs = ""
        raw_git_diffs = ""
        content = ""
    
    return {
        "model_patch": git_diffs.lstrip(),
        "raw_model_patch": raw_git_diffs.lstrip(),
        "original_file_content": content,
    }

def iterate_instance(
    test_spec: TestSpec,
    instance: dict,
    patch_id: int,
    container: docker.models.containers.Container,
    logger: logging.Logger,
    max_turns: int,
    cost_limit: float,
    test: str,
    edit: dict,
    model: str,
    temperature: float,
    initial_prompt: str,
    loc: dict,
    file_contents, file_loc_intervals,
    run_regression: bool,
    args: argparse.Namespace,
    oracle_mode: bool = False,
    init_passed_cases: list[str] = None,
    is_official_tests: bool = False
):
    instance_id = instance["instance_id"]
    instance_dir = instance["instance_dir"]
    
    logger.info(f"Iterating instance {instance_id} with patch #{patch_id}")
    
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0
    }
    cur_edit = edit['raw_edit']
    cur_patch = edit['model_patch']
    initial_trans_res = trans_res = {
        "model_patch": edit['model_patch'],
        "raw_model_patch": edit['raw_model_patch'],
        "original_file_content": edit['original_file_content']
    }
    
    report = {
        "log": [
            {
                "type": "initial status",
                "edit": cur_edit,
                "patch": cur_patch,
            }
        ],
        "usage": total_usage,
        "status": "running"
    }
    
    with open(instance_dir / f"{instance_id}_{patch_id}.json", "w") as f:
        json.dump(report, f, indent=4)
    
    for i in range(max_turns + 1):
        if cur_patch:
            test_result = run_test(patch_id, i, instance, container, test_spec, cur_patch, [test], logger, run_regression=False, run_libro=True, run_oracle=oracle_mode, is_official_tests=is_official_tests)
        else:
            test_result = {
                "success": "EDIT_WRONG",
            }
            
        report['log'].append({
            "type": "test",
            "result": test_result
        })
        
        with open(instance_dir / f"{instance_id}_{patch_id}.json", "w") as f:
            json.dump(report, f, indent=4)
        
        if test_result['success'] == "PASSED":
            report['status'] = "success"
            report['patch'] = cur_patch
            break
            
        if i == max_turns:
            report['status'] = "exit_turn"
            report['patch'] = cur_patch
            break
        if total_usage["cost"] > cost_limit:
            report['status'] = "cost_limit"
            report['patch'] = cur_patch
            break
            
        traj = query_llm(initial_prompt, cur_edit, test_result, model, temperature, logger)
        
        total_usage["completion_tokens"] += traj["usage"]["completion_tokens"]
        total_usage["prompt_tokens"] += traj["usage"]["prompt_tokens"]
        total_usage["cost"] += traj["usage"]["cost"]
        
        if not traj['stop']:
            cur_edit = traj['result']
            trans_res = trans_edit_to_patch(traj['response'], logger, traj, loc, file_contents, file_loc_intervals, args)
            cur_patch = trans_res['model_patch']
        report['log'].append({
            "type": "llm",
            "traj": traj,
            "edit": cur_edit,
            "patch": cur_patch,
            "trans_res": trans_res
        })
        
        with open(instance_dir / f"{instance_id}_{patch_id}.json", "w") as f:
            json.dump(report, f, indent=4)
        
        if traj['stop']:
            report['status'] = "stop_LLM"
            report['patch'] = edit['model_patch']
            cur_patch = edit['model_patch']
            trans_res = initial_trans_res
            report['log'].append(report['log'][1])
            break
    
    assert report['log'][-1]['type'] == "test"
    
    report['final_result'] = report['log'][-1]['result']
    if cur_patch == "":
        for log in reversed(report['log']):
            if log['type'] == "llm" and log['patch'] != "":
                cur_patch = log['patch']
                trans_res = log['trans_res']
                report['log'][-1]['true_patch'] = cur_patch
                break
        if cur_patch == "":
            cur_patch = edit['model_patch']
            trans_res = initial_trans_res
            report['log'][-1]['true_patch'] = cur_patch
    report['final_status'] = report['final_result']['success']

    if report['final_status'] != "PASSED":
        cur_patch = edit['model_patch']
        trans_res = initial_trans_res
        report['patch'] = cur_patch
        report['final_result'] = report['log'][1]['result']
        report['final_status'] = report['final_result']['success']
    
    if run_regression:
        test_result = run_test(patch_id, i + 1, instance, container, test_spec, cur_patch, [test], logger, run_regression=True, run_libro=False, regression_testcases=init_passed_cases)

        report['regression'] = {}
        report['regression']['result'] = test_result
        report['regression']['status'] = test_result['success']
        if test_result['success'] != "PASSED":
            cur_patch = edit['model_patch']
            trans_res = initial_trans_res
            report['status'] = "regression_failed"
            report['final_result'] = report['log'][1]['result']
            report['final_status'] = report['final_result']['success']
        
    if report['final_status'] == "PASSED" and report['status'] == 'success':
        report['final_libro_test_pass_num'] = 1
    # elif report['final_status'] == "LIBRO" or report['final_status'] == "EDIT_WRONG":
    else:
        report['final_libro_test_pass_num'] = 0
    # else:
    #     assert False, f"Unexpected final status: {report['final_status'], report['status']}"
        
    report['patch'] = cur_patch
    
    with open(instance_dir / f"{instance_id}_{patch_id}.json", "w") as f:
        json.dump(report, f, indent=4)
        
    return {
        "id": patch_id,
        "patch": cur_patch,
        "final_libro_test_pass_num": report['final_libro_test_pass_num'],
        "final_status": report['final_status'],
        "final_result": report['final_result'],
        "trans_res": trans_res
    }

def feedback_instance(
    test_spec: TestSpec,
    instance: dict,
    output_folder: str,
    output_file: str,
    model: str,
    temperature: float,
    skip_existing: bool,
    max_turns: int,
    cost_limit: float,
    existing_instance_ids: list[str],
    docker_client: docker.DockerClient,
    locs: list[dict],
    tests: list[dict],
    edits: list[dict],
    tests_type: str,
    run_regression: bool,
    args: argparse.Namespace,
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
        # for i, pred_file in enumerate(pred_files):
        #     if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
        #         file_to_edit_locs[pred_file] = loc["found_edit_locs"][i]
        if "found_edit_locs" in loc:
            file_to_edit_locs = loc["found_edit_locs"]

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
    
    # Build + start instance container (instance image should already be built)
    container = build_container(test_spec, docker_client, args.run_id, logger, False, False)
    container.start()
    logger.info(f"Container for {instance_id} started: {container.id}")
    
    if tests_type == "oracle":
        tests = [None]
        oracle_mode = True
        is_official_tests = False
    elif tests_type == "official":
        if 'test_patch' in tests:
            tests = [extract_new_file_content(tests['test_patch'])]
        else:
            tests = []
        oracle_mode = False
        is_official_tests = True
    else:
        tests = [ x['test'] for x in tests['final_tests']]
        if args.sort_tests:
            tests.sort(key=lambda x: len(x))
        tests = tests[:args.tests_top_n]
        oracle_mode = False
        is_official_tests = False
    edits = edits['edits'][:args.patches_top_n]
        
    # iter_result = {i: [] for i in range(-1, len(tests) + 1)}
    iter_result ={-1: [], 0: [], 1: []}
    
    try:
        if run_regression:
            init_report = run_test(-1, 0, instance, container, test_spec, "", [], logger, run_regression=True, run_libro=False, run_oracle=False)
            init_passed_cases = init_report['report'] if init_report['success'] == "REGRESSION" else init_report['report_regression']
            init_passed_cases = [k for k, v in init_passed_cases.items() if v == "PASSED"]
        else:
            init_passed_cases = []
        
        for i, edit in enumerate(edits):
            bl = edit['belong']
            for j, test in enumerate(tests):
                res = iterate_instance(
                    test_spec,
                    instance,
                    i * len(tests) + j,
                    container,
                    logger,
                    max_turns,
                    cost_limit,
                    test,
                    edit,
                    model,
                    temperature,
                    edit['prompt'],
                    locs[bl],
                    file_contentss[bl], file_loc_intervalss[bl],
                    run_regression, args,
                    oracle_mode = oracle_mode,
                    init_passed_cases = init_passed_cases,
                    is_official_tests = is_official_tests,
                )
                res['votes'] = edit['votes']
                iter_result[res['final_libro_test_pass_num']].append(res)
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
        pass
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
    
    # with open(output_file, "a") as f:
    #     f.write(json.dumps({
    #         "instance_id": instance_id,
    #         "all_patches": iter_result
    #     }) + "\n")
    #     logger.info(f"Final tests for {instance_id} written to {output_file}")
    
    # if iter_result[1]:
    #     iter_result[1].sort(key=lambda x: x['id'])
    #     model_patch = iter_result[1][0]['patch']
    #     patch_type = 1
    # elif iter_result[0]:
    #     iter_result[0].sort(key=lambda x: x['id'])
    #     model_patch = iter_result[0][0]['patch']
    #     patch_type = 0
    # else:
    #     model_patch = ""
    #     patch_type = -1
    
    top10 = []
    for i in [1, 0]:
        count_norm_patches = {}
        first_appear_idx = {}
        for res in iter_result[i]:
            try:
                normalized_patch = normalize_patch(
                    instance_id, res['patch'], res['trans_res']['original_file_content']
                ).strip()
            except Exception as e:
                with open("errorlog.txt", "a") as f:
                    f.write(f"Error in normalize_patch: ({instance_id}):\n {e}\n {res['patch']}\n")
                print(f"Error in normalize_patch: ({instance_id}):\n {e}\n {res['patch']}\n")
                logger.error(f"Error in normalize_patch: ({instance_id}):\n {e}\n {res['patch']}\n")
                normalized_patch = ""
            res['normalized_patch'] = normalized_patch
            if not normalized_patch:
                normalized_patch = res['patch'].strip()
            count_norm_patches[normalized_patch] = count_norm_patches.get(normalized_patch, 0) + res['votes']
            if normalized_patch not in first_appear_idx:
                first_appear_idx[normalized_patch] = res['id']
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
            if len(top10) >= 10:
                break
        if len(top10) >= 10:
            break
    
    while len(top10) < 10:
        top10.append({
            "patch": "",
            "patch_type": -1,
            "votes": 0
        })
    
    for i in range(10):
        with open(output_file if i == 0 else output_file.replace('.jsonl', f"_{i}.jsonl"), "a") as f:
            f.write(json.dumps({
                "model_name_or_path": "aglibro",
                "instance_id": instance_id,
                "patch_type": top10[i]['patch_type'],
                "votes": top10[i]['votes'],
                "model_patch": top10[i]['patch'],
            }) + "\n")
    logger.info(f"Final tests for {instance_id} written to {output_file}")

def feedback(
    dataset: str,
    split: str,
    output_folder: str,
    output_file: str,
    tests_file: str,
    loc_files: str,
    edits_file: str,
    model: str,
    temperature: float,
    num_threads: int,
    target_ids: list[str],
    skip_existing: bool,
    max_turns: int,
    cost_limit: float,
    tests_type: str,
    run_regression: bool,
    args: argparse.Namespace
):
    bench = load_dataset(dataset, split=split)
    if not target_ids:
        target_ids = [x["instance_id"] for x in bench]
    instances = [x for x in bench if x["instance_id"] in target_ids]
    existing_instance_ids = load_existing_instance_ids(output_file) if skip_existing else set()
    
    temp_instances = {
        instance['instance_id'] : {
            "model_name_or_path": model,
            "instance_id": instance['instance_id'],
            "model_patch": "<temp>"
        }
        for instance in instances
    }
    dataset = get_dataset_from_preds(dataset, split, target_ids, temp_instances, args.run_id)
    test_specs = list(map(make_test_spec, dataset))
    
    locs = dict()
    for loc_file in loc_files.split(','):
        loc = load_jsonl(loc_file)
        for instance in loc:
            locs.setdefault(instance['instance_id'], []).append(instance)
    all_edits = load_jsonl(edits_file)
    
    if tests_type != "oracle":
        all_tests = load_jsonl(tests_file)
        all_tests_ids = {edit['instance_id'] for edit in all_tests}
        for id in target_ids:
            if id not in all_tests_ids:
                all_tests.append({
                    "instance_id": id,
                    "final_tests": []
                })
    else:
        all_tests = [{"instance_id": id, "final_tests": "oracle"} for id in target_ids]
    
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
    if len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")
    
    # run reproduction in parallel
    print(f"Running {len(target_ids)} instances...")
    with tqdm(total=len(target_ids), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    feedback_instance,
                    test_spec,
                    [instance for instance in instances if instance["instance_id"] == test_spec.instance_id][0],
                    output_folder,
                    output_file,
                    model,
                    temperature,
                    skip_existing,
                    max_turns,
                    cost_limit,
                    existing_instance_ids,
                    client,
                    locs[test_spec.instance_id],
                    [tests for tests in all_tests if tests["instance_id"] == test_spec.instance_id][0],
                    [edit for edit in all_edits if edit["instance_id"] == test_spec.instance_id][0],
                    tests_type,
                    run_regression,
                    args
                ): None
                for test_spec in test_specs
                if test_spec.instance_id in locs and \
                    [tests for tests in all_tests if tests["instance_id"] == test_spec.instance_id] and \
                    [edit for edit in all_edits if edit["instance_id"] == test_spec.instance_id]
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
    parser.add_argument("--output_file", type=str, default="generated_patches.jsonl")
    parser.add_argument("--loc_files", type=str, required=True)
    parser.add_argument("--tests_file", type=str)
    parser.add_argument("--edits_file", type=str, required=True)
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        # choices=["gpt-4o", "deepseek-coder", "gpt-4o-mini"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
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
    
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--cost_limit", type=float, default=0.1)
    
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
    
    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    
    feedback(
        args.dataset,
        args.split,
        args.output_folder,
        args.output_file,
        args.tests_file,
        args.loc_files,
        args.edits_file,
        args.model,
        args.temperature,
        args.num_threads,
        args.target_ids,
        args.skip_existing,
        args.max_turns,
        args.cost_limit,
        args.tests_type,
        args.run_regression,
        args
    )

if __name__ == "__main__":
    main()