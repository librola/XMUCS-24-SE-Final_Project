import argparse
import concurrent.futures
import json
import os
import logging

from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from aglibro.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    setup_logger,
)
from aglibro.util.model import make_model

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)

TEMPLATE_DIR = 'aglibro/libro/prompt_templates/'

MAP_REPO_TO_ADDITIONAL_HINTS = {
    "sympy/sympy": "The test driver will run the functions that start with `test`, and if no exception is thrown during the execution of the function, the test passes. You don't need to use a testing framework like `pytest` or `unittest`.",
    "django/django": "Your test wiil be run by the test framework of django. You should provide a test case class that inherits from the base class `SimpleTestCase` or `TestCase` from the `django.test` module. Your test will be placed in the new `libro` directory under the `tests` directory and named `tests.py`. If you need to inherit from the `django.db.models.Model` class, make sure to include a `Meta` class in the model to explicitly declare the `app_label` attribute as `libro`."
}

def make_messages_from_file(problem_statement, repo, version, test_command, additional_hints,
                          template_file=TEMPLATE_DIR+'2example_chat.json'):

    with open(template_file) as f:
        messages = json.load(f)

        for msg in messages:
            example_text_path = re.findall(r'{%(.+?)%}', msg['content'])
            if len(example_text_path) > 0:
                for ef in example_text_path:

                    with open(os.path.join(TEMPLATE_DIR, ef)) as f:
                        example_text = f.read()
                    msg['content'] = msg['content'].replace('{%'+ef+'%}', example_text)

        current_query = messages[-1]['content']
        bug_report_content = f"""
        {problem_statement}
        """
        current_query = current_query.replace('{{bug_report_content}}', bug_report_content) \
                                     .replace('{{repo}}', repo) \
                                     .replace('{{version}}', version) \
                                     .replace('{{test_command}}', test_command) \
                                     .replace('{{additional_hints}}', additional_hints)

        messages[-1]['content'] = current_query

    return messages, None

def query_llm_for_gentest(bug, model, temperature, num_samples, template, logger):
    instance_id = bug["instance_id"]
    repo = bug["repo"]
    problem_statement = bug["problem_statement"]
    repo = bug["repo"]
    version = bug["version"]
    test_command = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    additional_hints = MAP_REPO_TO_ADDITIONAL_HINTS.get(repo, "")

    prompt, stop = make_messages_from_file(
        problem_statement,
        repo,
        version,
        test_command,
        additional_hints,
        template_file=TEMPLATE_DIR+f'{template}.json')

    logger.info(f"Prompt: {prompt}")

    model = make_model(
        model = model,
        logger = logger,
        backend = 'openai' if model.startswith('gpt') else ('deepseek' if model.startswith('deepseek') else 'claude'),
        temperature = temperature,
        max_tokens = 1024,
        batch_size = num_samples
    )
    trajs = model.codegen(prompt, num_samples=num_samples)
    
    for traj in trajs:
        query_result = traj["response"]
        if ("```python") in query_result:
            gen_test = query_result.split("```python")[1] \
                                   .split("```")[0]
        elif ("```") in query_result:
            gen_test = query_result.split("```")[1]
        else:
            gen_test = query_result
        traj["gen_test"] = gen_test
        traj["prompt"] = prompt
    
    return trajs

def generate_tests(
    instance: dict,
    logger: logging.Logger,
    output_file: Path,
    model: str,
    num_samples: int,
    temperature: float,
    skip_existing: bool,
    template: str,
    prev_output: list[dict] = None,
):
    instance_id = instance["instance_id"]
    
    logger.info(f"================ generating {instance_id} ================")
    
    matched_prev_o = None
    if skip_existing and prev_output:
        for o in prev_output:
            if o["instance_id"] == instance_id:
                matched_prev_o = o
                break
    if matched_prev_o:
        logger.info(f"Skipping generating for existing instance_id: {instance['instance_id']}")
        return matched_prev_o["gen_tests"], matched_prev_o["trajs"]
    
    trajs = query_llm_for_gentest(instance, model, temperature, num_samples, template, logger)
    gen_tests = [traj["gen_test"] for traj in trajs]
    logger.info(f"Generated {len(gen_tests)} tests: {gen_tests}")

    with open(output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance["instance_id"],
                    "gen_tests": gen_tests,
                    "trajs": trajs,
                }
            )
            + "\n"
        )
    return gen_tests, trajs

def generate_instance(
    instance, args, logger
):
    instance_id = instance["instance_id"]
    # log_file = os.path.join(args.output_folder, "generate_logs", f"{instance_id}.log")
    
    if args.target_id is not None:
        if instance["instance_id"] not in args.target_id:
            return

    # logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance["instance_id"] in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {instance['instance_id']}")
        return

    # if PROJECT_FILE_LOC is not None:
    #     project_file = os.path.join(PROJECT_FILE_LOC, bug["instance_id"] + ".json")
    #     d = load_json(project_file)
    # else:
    #     # we need to get the project structure directly
    #     d = get_project_structure_from_scratch(
    #         bug["repo"], bug["base_commit"], bug["instance_id"], "playground"
    #     )
    # 这几行是用来获得 repo 结构的，我们暂时应该还不需要

    logger.info(f"================ generating {instance_id} ================")

    problem_statement = instance["problem_statement"]
    # structure = d["structure"]
    
    trajs = query_llm_for_gentest(instance, args, args.template, logger)
    gen_tests = [traj["gen_test"] for traj in trajs]
    logger.info(f"Generated {len(gen_tests)} tests: {gen_tests}")

    # filter_none_python(structure)  # some basic filtering steps
    # 这句话应该是过滤掉不是 python 代码的文件，暂时应该不到，不过之后应该会很有用

    # # filter out test files (unless its pytest)
    # if not d["instance_id"].startswith("pytest"):
    #     filter_out_test_files(structure)

    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance["instance_id"],
                    "gen_tests": gen_tests,
                    "trajs": trajs,
                }
            )
            + "\n"
        )


def generate(args):
    bench_data = load_dataset(args.dataset, split=args.split)
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bug in bench_data:
            generate_instance(
                bug, args, existing_instance_ids
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    generate_instance,
                    bug,
                    args,
                    existing_instance_ids,
                )
                for bug in bench_data
            ]
            concurrent.futures.wait(futures)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="generated_tests.jsonl")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-4o", "deepseek-coder", "gpt-4o-mini"],
    )
    parser.add_argument('--template', default='2example_chat')
    # parser.add_argument('--save_prompt', action='store_true')
    # parser.add_argument('--prompt_save_path', type=str, default=None)

    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)

    assert (
        not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "generate_logs"), exist_ok=True)
    
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        
    args.output_file = os.path.join(args.output_folder, "generated_tests.jsonl")
        
    generate(args)

    # # write the arguments
    # with open(f"{args.output_folder}/args.json", "w") as f:
    #     json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    main()
    