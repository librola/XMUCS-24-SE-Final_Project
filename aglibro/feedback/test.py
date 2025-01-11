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
    parse_test_output_to_error
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
    TestStatus,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec, TestSpec, DIFF_MODIFIED_FILE_REGEX
from swebench.harness.utils import load_swebench_dataset, str2bool, get_test_directives
from swebench.harness.run_evaluation import EvaluationError, get_dataset_from_preds

from aglibro.libro.llm_prompt import generate_tests
from aglibro.libro.postprocess import run_generate_test
from aglibro.libro.llm_regenerate import regenerate_tests

MAP_REPO_TO_TEST_ARG = {
    "django/django": "libro.tests",
    "sympy/sympy": "sympy/libro/tests/test_libro.py",
    "default": "/tmp/test_libro.py",
}
MAP_REPO_TO_TEST_PATH = {
    "django/django": "/testbed/tests/libro/tests.py",
    "sympy/sympy": "/testbed/sympy/libro/tests/test_libro.py",
    "default": "/tmp/test_libro.py",
}

def test_passed(case: str, sm: dict[str, str]) -> bool:
    # return case not in sm or sm[case] == TestStatus.PASSED.value or sm[case] == TestStatus.SKIPPED.value
    return case in sm and sm[case] == TestStatus.PASSED.value
def all_tests_passed(cases: list[str], sm: dict[str, str]) -> bool:
    return not cases or (any(case in sm for case in cases) and all(test_passed(case, sm) for case in cases))
def instance_regression_check(instance: dict, sm: dict[str, str]) -> bool:
    return all_tests_passed(eval(instance['PASS_TO_PASS']), sm)

def update_test_command(command):
    if command.startswith('pytest'):
        # 替换 --tb=no 为 --tb=long
        command = command.replace('--tb=no', '--tb=short')
        command = command.replace('--tb=long', '--tb=short')
        command = command.replace('--tb=auto', '--tb=short')
        
        # 如果命令中没有指定 --tb 参数，并且是 pytest 命令，则添加 --tb=long
        if '--tb=' not in command:
            command = command.replace('pytest', 'pytest --tb=short')
        
        # 为确保所有 pytest 命令都包含 -rA 选项，如果没有的话添加上
        if '-rA' not in command:
            command = command.replace('pytest', 'pytest -rA')
    
    if command.startswith('tox'):
        command = command.replace('-v --', '-v -- -q --disable-warnings -s')
    
    return command

def make_test_script_list(instance, specs, env_name, repo_directory, base_commit, edit_patch, test_command_update=False, regressed=False, test_patch="", is_official_tests=False):
    """
    Applies the test patch and runs the tests.
    """
    if edit_patch:
        HEREDOC_DELIMITER = "EOF_114329324912"
        edit_patch_files = re.findall(DIFF_MODIFIED_FILE_REGEX, edit_patch)
        # Reset test files to the state they should be in before the patch.
        apply_edit_patch_command = (
            f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{edit_patch}\n{HEREDOC_DELIMITER}"
        )
    else:
        apply_edit_patch_command = ""
    
    if test_patch:
        HEREDOC_DELIMITER = "EOF_998244353"
        test_patch_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
        # Reset test files to the state they should be in before the patch.
        # reset_test_patch_command = f"git checkout {base_commit} {' '.join(test_patch_files)}"
        apply_test_patch_command = (
            f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
    else:
        # reset_test_patch_command = ""
        apply_test_patch_command = ""
    
    patch_files = list(set((edit_patch_files if edit_patch else []) + (test_patch_files if test_patch else [])))
    reset_patch_command = f"git checkout {base_commit} {' '.join(patch_files)}"
    
    if is_official_tests:
        test_command = "python " + MAP_REPO_TO_TEST_PATH.get(instance["repo"], MAP_REPO_TO_TEST_PATH["default"])
    else:
        test_cmd_pre = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
        if test_command_update:
            test_cmd_pre = update_test_command(test_cmd_pre)
        if regressed:
            test_args = get_test_directives(instance)
        else:
            test_args = [MAP_REPO_TO_TEST_ARG.get(instance["repo"], MAP_REPO_TO_TEST_ARG["default"])]
        test_command = " ".join(
            [
                test_cmd_pre,
                # *get_test_directives(instance),
                *test_args
            ]
        )
    
    eval_commands = [
        f"source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        # f"git status",
        # f"git show",
        # f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_patch_command,
        apply_edit_patch_command,
        apply_test_patch_command,
        test_command,
        reset_patch_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands, test_command, reset_patch_command

def make_test_script(instance, specs, env_name, repo_directory, base_commit, edit_patch, test_command_update=False, regressed=False, test_patch="", is_official_tests=False):
    a, b, c = make_test_script_list(instance, specs, env_name, repo_directory, base_commit, edit_patch, test_command_update, regressed, test_patch, is_official_tests)
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + a) + "\n", b, c

def run_test(
    patch_id: int,
    turn_id: int,
    instance: dict,
    container: docker.models.containers.Container,
    test_spec: TestSpec,
    patch: str,
    tests: list[str],
    # test_command: str,
    logger: logging.Logger,
    run_oracle: bool = False,
    run_regression: bool = True,
    run_libro: bool = True,
    regression_testcases: list[str] = None,
    is_official_tests: bool = False
):
    instance_id = instance["instance_id"]
    instance_dir = instance["instance_dir"]
    
    logger.info(f"================ Running {instance_id} patch #{patch_id} turn #{turn_id} ================")
    
    repo = instance["repo"]
    version = instance["version"]
    spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
    timeout = 300
    
    patch_file = instance_dir / f"patch_{patch_id}_{turn_id}.patch"
    patch_file.write_text(patch)
    
    if regression_testcases == None:
        regression_testcases = eval(instance['PASS_TO_PASS'])
    
    if run_oracle:
        test_patch = instance["test_patch"]
        test_script_oracle = instance_dir / "do_test_oracle.sh"
        test_script_oracle_content, test_command_oracle, reset_command_oracle = make_test_script(instance, spec, "testbed", "/testbed", instance["base_commit"], patch, True, regressed=True, test_patch=test_patch)
        test_script_oracle.write_text(test_script_oracle_content)
        copy_to_container(container, test_script_oracle, Path("/do_test_oracle.sh"))
        logger.info(f"Test script for {instance_id} written to {test_script_oracle}, copied to container as /do_test_oracle.sh")
        
        # regression_test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test_regression.sh", timeout)
        oracle_test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test_oracle.sh", timeout)
        test_output_path = instance_dir / f"test_output_{patch_id}_{turn_id}_oracle.txt"
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path, "w") as f:
            f.write(oracle_test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                logger.info(f"Test timed out after {timeout} seconds.")
        
        report_oracle = get_logs_eval(test_output_path)
        f2p_list = eval(instance['FAIL_TO_PASS'])
        oracle_test_passed = all_tests_passed(f2p_list, report_oracle)
        error_info = parse_test_output_to_error(f2p_list, oracle_test_output)
        report_oracle = {k: v for k, v in report_oracle.items() if k in f2p_list}
        
        if not oracle_test_passed and not report_oracle:
            report_oracle[f2p_list[0]] = "ERROR"
            error_info[f2p_list[0]] = parse_output(oracle_test_output, test_command_oracle, reset_command_oracle)
        
        if not oracle_test_passed:
            return {
                "success": "ORACLE",
                "output": parse_output(oracle_test_output, test_command_oracle, reset_command_oracle),
                "report": report_oracle,
                "error_info": error_info,
                "test_command": test_command_oracle,
                "test_patch": test_patch,
            }
        else:
            return { "success": "PASSED" }
    
    # Copy test script to container
    if run_regression:
        test_script_regression = instance_dir / "do_test_regression.sh"
        test_script_regression_content, test_command_regression, reset_command_regression = make_test_script(instance, spec, "testbed", "/testbed", instance["base_commit"], patch, True, regressed=True)
        test_script_regression.write_text(test_script_regression_content)
        copy_to_container(container, test_script_regression, Path("/do_test_regression.sh"))
        logger.info(f"Test script for {instance_id} written to {test_script_regression}, copied to container as /do_test_regression.sh")
        
    
    if run_libro:
        test_script_libro = instance_dir / "do_test_libro.sh"
        spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
        test_script_libro_content, test_command_libro, reset_command_libro = make_test_script(instance, spec, "testbed", "/testbed", instance["base_commit"], patch, True, regressed=False, is_official_tests=is_official_tests)
        test_script_libro.write_text(test_script_libro_content)
        copy_to_container(container, test_script_libro, Path("/do_test_libro.sh"))
        logger.info(f"Test script for {instance_id} written to {test_script_libro}, copied to container as /do_test_libro.sh")
    
    # Run regression eval script, write output to logs
    if run_regression:
        regression_test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test_regression.sh", timeout)
        test_output_path = instance_dir / f"test_output_{patch_id}_{turn_id}_regression.txt"
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path, "w") as f:
            f.write(regression_test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                logger.info(f"Test timed out after {timeout} seconds.")
        
        report_regression = get_logs_eval(test_output_path)
        # regression_test_passed = set(report_regression.values()) == {"PASSED"} or set(report_regression.values()) == {"SKIPPED"} or set(report_regression.values()) == {"SKIPPED", "PASSED"}
        # regression_test_passed = instance_regression_check(instance, report_regression)
        regression_test_passed = all_tests_passed(regression_testcases, report_regression)
        logger.info(f"Grading report: {report_regression}\nRegression test {'PASSED' if regression_test_passed else 'FAILED'}")
        
        parsed_regression_test_output = parse_output(regression_test_output, test_command_regression, reset_command_regression)
        if not regression_test_passed:
            return {
                "success": "REGRESSION",
                "output": parsed_regression_test_output,
                "report": report_regression,
                "test_command": test_command_regression,
            }
    
    if run_libro:
    
        libro_res = []
        for i, test in enumerate(tests):
            logger.info(f"Running patch #{patch_id} turn #{turn_id} test #{i} ================")
            
            # Write test file to instance directory and copy to container
            libro_test_file = Path(instance_dir / f"libro_test_{i}.py")
            # print(test)
            libro_test_file.write_text(test)
            logger.info(
                f"Intermediate test for {instance_id} written to {libro_test_file}, now applying to container..."
            )
            target_path = MAP_REPO_TO_TEST_PATH.get(repo, MAP_REPO_TO_TEST_PATH["default"])
            copy_to_container(container, libro_test_file, Path(target_path))
            logger.info(f"Test file copied to container: {target_path}")
            
            # Run eval script, write output to logs
            libro_test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test_libro.sh", timeout)
            test_output_path = instance_dir / f"test_output_{patch_id}_{turn_id}_test_{i}.txt"
            logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
            with open(test_output_path, "w") as f:
                f.write(libro_test_output)
                logger.info(f"Test output for {instance_id} written to {test_output_path}")
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    logger.info(f"Test timed out after {timeout} seconds.")
                    
            if not is_official_tests:
                report = get_logs_eval(test_output_path)
                test_passed = set(report.values()) == {"PASSED"} or set(report.values()) == {"SKIPPED"} or set(report.values()) == {"SKIPPED", "PASSED"}
            else:
                expected_output = "Issue resolved"
                other_patterns = ["Issue reproduced", "Other issues"]
                test_passed = True
                for pattern in other_patterns:
                    if pattern in libro_test_output:
                        test_passed = False
                        break
                if expected_output not in libro_test_output:
                    test_passed = False
                report = {"resolved": test_passed}
            logger.info(f"Grading report: {report}\nLibro Test {'PASSED' if test_passed else 'FAILED'}")
            
            parsed_libro_test_output = parse_output(libro_test_output, test_command_libro, reset_command_libro)
            if not test_passed:
                libro_res.append({
                    "success": "LIBRO",
                    "test_id": i,
                    "test": test,
                    "output": parsed_libro_test_output,
                    "report": report,
                    "test_command": test_command_libro,
                    "reset_command": reset_command_libro
                })
        
        if len(libro_res) > 0:
            return {
                "success": "LIBRO",
                "res": libro_res,
            }
    
    return {
        "success": "PASSED",
        "report_regression": report_regression if run_regression else None
    }