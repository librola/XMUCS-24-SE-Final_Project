import argparse
import concurrent.futures
import json
import os

from datasets import load_dataset
from tqdm import tqdm

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
from aglibro.util.postprocess_tests import get_logs_eval_with_repo

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)

TEMPLATE_DIR = 'aglibro/libro/prompt_templates/'

def make_messages_from_file(problem_statement, repo, version, test, output,
                          template_file=TEMPLATE_DIR+'llm_check_prompt.json'):

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
                                     .replace('{{test_file_content}}', test) \
                                     .replace('{{test_output}}', output) \
                                     .replace('{{repo}}', repo) \
                                     .replace('{{version}}', version)

        messages[-1]['content'] = current_query

    return messages, None

def query_llm_for_check(instance_output, bench_data, args, template, logger):
    instance_id = bench_data["instance_id"]
    repo = bench_data["repo"]
    version = bench_data["version"]
    problem_statement = bench_data["problem_statement"]
    test = instance_output['test']
    output = instance_output['output']

    prompt, stop = make_messages_from_file(
        problem_statement,
        repo,
        version,
        test,
        output,
        template_file=TEMPLATE_DIR+f'{template}.json')
                
    logger.info(f"Prompt: {prompt}")

    model = make_model(
        model = args.model,
        logger = logger,
        backend = 'openai' if args.model.startswith('gpt') else 'deepseek',
        temperature = args.temperature,
        max_tokens = 1024,
        batch_size = 1
    )
    trajs = model.codegen(prompt, num_samples=1)
    
    print(trajs)
    
    traj = trajs[0]
    query_result = traj["response"].strip()
    if query_result.split('\n')[-1].strip().lower() == 'yes':
        success = True
        reason = "\n".join(query_result.split('\n')[:-1])
    elif query_result.lower().endswith('yes'):
        success = True
        reason = query_result[:-3].strip()
    elif query_result.split('\n')[-1].strip().lower() == 'no':
        success = False
        reason = "\n".join(query_result.split('\n')[:-1])
    elif query_result.lower().endswith('no'):
        success = False
        reason = query_result[:-2].strip()
    else:
        success = False
        reason = query_result
    
    traj["success"] = success
    traj["reason"] = reason
    
    return traj

def check_instance(
    instance_outputs, args, swe_bench_data, existing_instance_ids
):
    instance_id = instance_outputs["instance_id"]
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    log_file = os.path.join(args.output_folder, "generate_logs", f"{instance_id}.log")
    
    if args.target_id is not None:
        if instance_outputs["instance_id"] not in args.target_id:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance_outputs["instance_id"] in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {instance_outputs['instance_id']}")
        return

    logger.info(f"================ checking {instance_id} ================")

    problem_statement = bench_data["problem_statement"]
    test_outputs = instance_outputs["outputs"]
    
    final_tests = []
    for test_output in test_outputs:
        traj = query_llm_for_check(test_output, bench_data, args, args.template, logger)
        is_success = traj["success"]
        logger.info(f"Instance {instance_id} test #{test_output['index']} is_success: {is_success}\nReason: {traj['reason']}\n")
        if 'resolved' in test_output:
            resolved_before = test_output['resolved']
        else:
            report_before = get_logs_eval_with_repo(test_output["raw_output"], bench_data["repo"])
            resolved_before = set(report_before.values()) == {"PASSED"}
            
        if is_success:
            final_tests.append({
                "test": test_output['test'],
                "resolved_before": resolved_before,
            })
        test_output['success'] = is_success
        test_output['reason'] = traj['reason']
        test_output['traj'] = traj

    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_outputs["instance_id"],
                    "final_tests": final_tests,
                }
            )
            + "\n"
        )
    with open(f"{args.output_folder}/full_result.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_outputs["instance_id"],
                    "tests": test_outputs,
                }
            )
            + "\n"
        )

def llm_check(args):
    bench_data = load_dataset(args.dataset, split=args.split)
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )
    test_outputs = load_jsonl(args.test_outputs_file)

    if args.num_threads == 1:
        for instance_outputs in test_outputs:
            check_instance(
                instance_outputs, args, bench_data, existing_instance_ids
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    check_instance,
                    instance_outputs,
                    args,
                    bench_data,
                    existing_instance_ids,
                )
                for instance_outputs in test_outputs
            ]
            concurrent.futures.wait(futures)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="final_tests.jsonl")
    parser.add_argument("--test_outputs_file", type=str, required=True)
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
    parser.add_argument('--template', default='llm_check_prompt')
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
        
    llm_check(args)

    # # write the arguments
    # with open(f"{args.output_folder}/args.json", "w") as f:
    #     json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    main()
    