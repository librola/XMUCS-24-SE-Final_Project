import argparse
import concurrent.futures
import json
import os
import logging
import copy

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

def make_messages_from_file(messages, prev_output, resolved,
                          template_file=TEMPLATE_DIR+'regenerate_prompt.json'):

    with open(template_file) as f:
        msg = json.load(f)

        example_text_path = re.findall(r'{%(.+?)%}', msg['content'])
        if len(example_text_path) > 0:
            for ef in example_text_path:
                with open(os.path.join(TEMPLATE_DIR, ef)) as f:
                    example_text = f.read()
                msg['content'] = msg['content'].replace('{%'+ef+'%}', example_text)

        current_query = msg['content']
        current_query = current_query.replace('{{output}}', prev_output) \
                                     .replace('{{is_passed}}', "passed" if resolved else "didn't passed")
        msg['content'] = current_query
        messages.append(msg)

    return messages

def query_llm_for_gentest(instance, messages, prev_output, prev_resolved, model, temperature, template, logger):
    instance_id = instance["instance_id"]
    

    prompt = make_messages_from_file(
        messages,
        prev_output,
        prev_resolved,
        template_file=TEMPLATE_DIR+f'{template}.json')

    logger.info(f"Prompt: {prompt}")

    model = make_model(
        model = model,
        logger = logger,
        backend = 'openai' if model.startswith('gpt') else 'deepseek',
        temperature = temperature,
        max_tokens = 1024,
        batch_size = 1
    )
    trajs = model.codegen(prompt, num_samples=1)
    assert len(trajs) == 1, f"Expected 1 trajectory, got {len(trajs)}"
    
    traj = trajs[0]
    query_result = traj["response"].strip()
    if query_result.split('\n')[0].strip().lower() == 'yes':
        success = True
        output = "\n".join(query_result.split('\n')[1:])
    elif query_result.lower().startswith('yes'):
        success = True
        output = query_result[3:].strip()
    elif query_result.split('\n')[0].strip().lower() == 'no':
        success = False
        output = "\n".join(query_result.split('\n')[1:])
    elif query_result.lower().startswith('no'):
        success = False
        output = query_result[2:].strip()
    else:
        success = False
        output = query_result
    
    traj['success'] = success
    if success:
        traj['reason'] = output
    else:
        if ("```python") in output:
            gen_test = output.split("```python")[1] \
                                    .split("```")[0]
        elif ("```") in output:
            gen_test = output.split("```")[1]
        else:
            gen_test = output
        traj["test"] = gen_test
    
    return trajs

def regenerate_tests(
    test_id: int,
    turn_id: int,
    instance: dict,
    logger: logging.Logger,
    prev_output: str,
    prev_resolved: bool,
    messages: list[dict],
    traj: list[dict],
    model: str,
    temperature: float,
    template: str,
):
    instance_id = instance["instance_id"]
    instance_dir = instance["instance_dir"]
    
    logger.info(f"================ Regenerating {instance_id} test #{test_id} turn #{turn_id} ================")
    
    trajs = query_llm_for_gentest(instance, messages, prev_output, prev_resolved, model, temperature, template, logger)
    res = trajs[0]
    
    messages.append({
        "role": "assistant",
        "content": res["response"]
    })
    traj.extend(copy.deepcopy(messages[-2:]))
    traj[-1]["usage"] = res["usage"]
    traj[-1]["success"] = res["success"]
    if res["success"]:
        traj[-1]["reason"] = res["reason"]
    else:
        traj[-1]["test"] = res["test"]
    
    return
    

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
    