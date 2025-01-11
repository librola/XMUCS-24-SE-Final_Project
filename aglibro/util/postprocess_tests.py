import tokenize
from io import BytesIO
from pathlib import Path

import re

from swebench.harness.constants import (
    MAP_REPO_VERSION_TO_SPECS,
    APPLY_PATCH_PASS
)
from swebench.harness.utils import (
    get_test_directives,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"

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

def get_tokens(code_lines):
    """从代码行列表中提取所有的 token 集合"""
    code = "\n".join(code_lines)
    code_bytes = BytesIO(code.encode('utf-8'))
    tokens = {token.string for token in tokenize.tokenize(code_bytes.readline) if token.type == tokenize.NAME}
    return tokens

def calculate_similarity(tokens_t, tokens_c):
    """根据公式计算两个 token 集合的相似度"""
    intersection = tokens_t.intersection(tokens_c)
    return len(intersection) / len(tokens_t) if tokens_t else 0

def find_most_similar_class(test_method, classes):
    tokens_t = get_tokens(test_method.splitlines())
    max_similarity = -1
    most_similar_class = None

    for class_info in classes:
        class_name = class_info["name"]
        class_code_lines = class_info["text"]
        tokens_c = get_tokens(class_code_lines)
        similarity = calculate_similarity(tokens_t, tokens_c)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_class = class_name

    return most_similar_class, max_similarity

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

def make_test_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch, test_command_update=False, is_official_test=False):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    # test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    # reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    reset_tests_command = f""
    # apply_test_patch_command = (
    #     f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    # )
    if is_official_test:
        test_command = "python " + MAP_REPO_TO_TEST_PATH.get(instance["repo"], MAP_REPO_TO_TEST_PATH["default"])
    else:
        test_cmd_pre = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
        if test_command_update:
            test_cmd_pre = update_test_command(test_cmd_pre)
        test_command = " ".join(
            [
                test_cmd_pre,
                # *get_test_directives(instance),
                MAP_REPO_TO_TEST_ARG.get(instance["repo"], MAP_REPO_TO_TEST_ARG["default"]),
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
        f"git status",
        f"git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        # apply_test_patch_command,
        test_command,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands, test_command

def make_test_script(instance, specs, env_name, repo_directory, base_commit, test_patch, test_command_update=False, is_official_test=False):
    a, b = make_test_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch, test_command_update, is_official_test)
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + a) + "\n", b

def parse_output(output, command, reset_command=""):
    if command.startswith("tox"):
        output = output.split(command + "\n")[-1].strip()
        output = output.split("The ePub file is in")[-1]
        output = "\n".join(output.split("\n")[1:])
    if command.startswith("PYTHONWARNINGS"):
        command = command.replace("PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' ", "")
        
    output = output.split("+ " + command + "\n")[-1].strip()
    if reset_command and reset_command in output:
        output = output.split(reset_command)[0].strip()
    return output

def get_logs_eval_with_repo(content: str, repo: str) -> dict[str, str]:
    log_parser = MAP_REPO_TO_PARSER[repo]

    # Get status map of evaluation results
    content = content.split(f"{APPLY_PATCH_PASS} (pred)")[-1]
    if repo == "sympy/sympy":
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.endswith("[OK]") or line.endswith("[PASS]") or line.endswith("[FAIL]"):
                end = '[' + line.split("[")[-1].strip()
                lines[i] = line.split(end)[0].strip() + "\n" + end
        content = "\n".join(lines)
    return log_parser(content)

def get_logs_eval(log_fp: str) -> dict[str, str]:
    # Convert e.g. "logs/scikit-learn__scikit-learn-12421/test_output.txt" to "scikit-learn/scikit-learn"
    sample_id = str(Path(log_fp).parent.stem)  # e.g. scikit-learn__scikit-learn-12421
    repo = "-".join(sample_id.replace("__", "/").split("-")[:-1])  # e.g. scikit-learn/scikit-learn
    # log_parser = MAP_REPO_TO_PARSER[repo]

    with open(log_fp) as f:
        content = f.read()
        return get_logs_eval_with_repo(content, repo)
    
def parse_error_info_pytest(log_content: str):
    # lines = log_content.split('\n')
    # error_infos = [line[4: ] for line in lines if line.startswith('E   ')]
    # error_info = "\n".join(error_infos)
    # return error_info
    lines = log_content.splitlines()

    errors = []
    current_error = []
    
    for line in lines:
        if line.startswith("E   "):
            current_error.append(line)
        else:
            if current_error:
                errors.append("\n".join(current_error))
                current_error = []
    
    if current_error:
        errors.append("\n".join(current_error))

    return errors[0] if errors else ""
    
def parse_error_info_django(log_content: str):
    error_pattern = r"(?s)([\w.]+(?:Error|Warning|Forbidden): .+?)(?=\n\n----------------------------------------------------------------------\n)"
    errors = re.findall(error_pattern, log_content)
    return errors[0] if errors else ""

def parse_error_info_sympy(log_content: str):
    lines = log_content.splitlines()
    
    errors = []
    current_error = []
    in_error_block = False
    
    for line in lines:
        if re.match(r"[\w.]*Error", line) or re.match(r"[\w.]*Exception", line) or re.match(r"[\w.]*Warning: ", line):
            if in_error_block:
                current_error.append(line)
            else:
                in_error_block = True
                current_error = [line]
        elif in_error_block:
            if line.strip() == "":
                errors.append("\n".join(current_error))
                in_error_block = False
                current_error = []
            else:
                current_error.append(line)

    if in_error_block and current_error:
        errors.append("\n".join(current_error))
    
    return errors[0]

def parse_error_sphinx(log_content: str):
    return parse_error_info_pytest(log_content)

def parse_error_info(log_content: str, repo: str, version: str):
    test_command = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if "pytest" in test_command:
        return parse_error_info_pytest(log_content)
    elif repo == "django/django":
        return parse_error_info_django(log_content)
    elif repo == "sympy/sympy":
        return parse_error_info_sympy(log_content)
    elif repo == "sphinx-doc/sphinx":
        return parse_error_sphinx(log_content)
    else:
        raise ValueError(f"Unknown repo: {repo}")

def parse_test_output_to_error(testcases, output):
    testcase_errors = { testcase: "" for testcase in testcases }

    lines = output.splitlines()

    current_testcase = None
    current_error_lines = []

    for line in lines:
        stripped_line = line.strip()

        if any(testcase in stripped_line for testcase in testcases):
            if current_testcase is not None:
                testcase_errors[current_testcase] = "\n".join(current_error_lines).strip()

            current_testcase = [testcase for testcase in testcases if testcase in stripped_line][0]
            current_error_lines = []

        elif current_testcase:
            current_error_lines.append(line)

    if current_testcase is not None:
        testcase_errors[current_testcase] = "\n".join(current_error_lines).strip()

    return testcase_errors

def extract_new_file_content(patch: str) -> str:
    # 使用正则表达式匹配新文件内容的起始部分
    content_start = re.search(r'\+\+\+ b/reproduce_bug\.py\n(@@.*?@@\n)?', patch)
    
    # 确认找到了文件内容的起始位置
    if content_start:
        start_index = content_start.end()
        # 提取所有以 '+' 开头的行，代表新文件的内容
        content_lines = [
            line[1:] for line in patch[start_index:].splitlines()
            if line.startswith('+') and not line.startswith('+++')
        ]
        # 返回拼接后的文件内容
        return '\n'.join(content_lines)
    
    # 如果没有找到内容，返回空字符串
    return ""