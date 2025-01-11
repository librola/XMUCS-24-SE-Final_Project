import argparse
import concurrent.futures
import json
import os
from difflib import unified_diff
import docker
import traceback

from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from aglibro.util.api_requests import num_tokens_from_messages
from aglibro.util.model import make_model
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
from aglibro.util.utils import load_jsonl
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    # INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
    MAP_REPO_VERSION_TO_SPECS
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
    INSTANCE_IMAGE_BUILD_DIR
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool
from swebench.harness.run_evaluation import EvaluationError, get_dataset_from_preds
from aglibro.util.postprocess_tests import make_test_script, parse_output, get_logs_eval, MAP_REPO_TO_TEST_PATH
from swebench.harness.constants import (
    PASS_TO_PASS,
    PASS_TO_FAIL,
    FAIL_TO_PASS,
    FAIL_TO_FAIL,
)

RUNNING_FAILED = "RUNNING_FAILED"

def map_resolved_to_pass_fail(resolved_before, resolved_after, running_failed):
    if running_failed:
        return RUNNING_FAILED
    elif resolved_before and resolved_after:
        return PASS_TO_PASS
    elif resolved_before and not resolved_after:
        return PASS_TO_FAIL
    elif not resolved_before and resolved_after:
        return FAIL_TO_PASS
    elif not resolved_before and not resolved_after:
        return FAIL_TO_FAIL
    else:
        raise ValueError("Invalid resolved_before and resolved_after values")

TEST_EVAL_LOG_DIR = Path("results/test_eval")

test_info_mapping = {
    "astropy": {
        "path": "-",
        "type": "function"
    },
    "django": {
        "path": "tests/",
        "type": "method",
    },
    "matplotlib": {
        "path": "lib/matplotlib/tests",
        "type": "function",
    },
    "mwaskom/seaborn": {
        "path": "tests/",
        "type": "method",
    },
    "pallets/flask": {
        "path": "tests/",
        "type": "function",
    },
    "psf/requests": {
        "path": ["tests/", "test_requests.py"],
        "type": ["method", "function"],
    },
    "pydata/": {
        "path": "xarray/tests/",
        "type": ["function"]
    }
}

report = {
    "total_instance_count": 0,
    "shortest_all": {
        "FAIL_TO_PASS_count": 0,
        "FAIL_TO_FAIL_count": 0,
        "PASS_TO_FAIL_count": 0,
        "PASS_TO_PASS_count": 0,
        "RUNNING_FAILED_count": 0,
        "FAIL_TO_PASS": [],
        "FAIL_TO_FAIL": [],
        "PASS_TO_FAIL": [],
        "PASS_TO_PASS": [],
        "RUNNING_FAILED": [],
    },
    "shortest_failed_before": {
        "FAIL_TO_PASS_count": 0,
        "FAIL_TO_FAIL_count": 0,
        "PASS_TO_FAIL_count": 0,
        "PASS_TO_PASS_count": 0,
        "RUNNING_FAILED_count": 0,
        "FAIL_TO_PASS": [],
        "FAIL_TO_FAIL": [],
        "PASS_TO_FAIL": [],
        "PASS_TO_PASS": [],
        "RUNNING_FAILED": [],
    },
    "exist_f2p_count": 0,
    "top1_f2p_count": 0,
    "top2_f2p_count": 0,
    "top3_f2p_count": 0,
    "total_tests_all": {
        "total_tests_count": 0,
        "FAIL_TO_PASS_count": 0,
        "FAIL_TO_FAIL_count": 0,
        "PASS_TO_FAIL_count": 0,
        "PASS_TO_PASS_count": 0,
        "RUNNING_FAILED_count": 0,
    },
    "total_tests_failed_before": {
        "total_tests_count": 0,
        "FAIL_TO_PASS_count": 0,
        "FAIL_TO_FAIL_count": 0,
        "PASS_TO_FAIL_count": 0,
    },
}

def get_instance_image_key(instance):
    return f"sweb.eval.x86_64.{instance['instance_id']}:latest"

def _post_process_multifile_repair(
    raw_output: str,
    file_contents: dict[str, str],
    logger,
    file_loc_intervals: dict[str, list],
    diff_format=False,
):
    edit_multifile_commands = extract_python_blocks(raw_output)
    edited_file = ""
    new_content = ""
    try:
        file_to_commands = split_edit_multifile_commands(
            edit_multifile_commands, diff_format=diff_format
        )
        logger.info("=== file_to_commands: ===")
        logger.info(json.dumps(file_to_commands, indent=2))
        # Let's only edit the first file in the edit commands.
        edited_file_key = next(iter(file_to_commands.keys()))
        logger.info(f"=== edited_file: {edited_file_key} ===")
        edit_commands = file_to_commands[edited_file_key]
        logger.info("=== edit_commands: ===")
        for c in edit_commands:
            logger.info(c)
            logger.info("\n" + "-" * 40)
        edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
        content = file_contents[edited_file]
        if diff_format:
            new_content = parse_diff_edit_commands(
                edit_commands, content, file_loc_intervals[edited_file]
            )
        else:
            new_content = parse_edit_commands(edit_commands, content)
    except Exception as e:
        logger.error(e)
        return edited_file, new_content

    diff = list(
        unified_diff(
            content.split("\n"),
            new_content.split("\n"),
            fromfile=edited_file,
            tofile=edited_file,
            lineterm="",
        )
    )

    logger.info(f"extracted patch:")
    logger.info("\n".join(diff))
    print("\n".join(diff))
    return edited_file, new_content


def construct_topn_file_context(
    file_to_locs,
    pred_files,
    file_contents,
    structure,
    context_window: int,
    loc_interval: bool = True,
    fine_grain_loc_only: bool = False,
    add_space: bool = False,
    sticky_scroll: bool = False,
    no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals


def instance_tests_eval(tests, test_spec : TestSpec, args, swe_bench_data, prev_o, client):
    instance_id = tests["instance_id"]
    log_dir = Path(args.output_folder) / instance_id
    
    # log_file = os.path.join(
    #     args.output_folder, instance_id, f"{instance_id}.log"
    # )
    log_file = Path(args.output_folder) / instance_id / f"{instance_id}.log"
    logger = setup_logger(instance_id, log_file)
    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    if found:
        logger.info(f"skipping {instance_id} since patch already generated")
        return None
    
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    logger.info(f"================ evaluating {instance_id} ================")
    if len(tests["final_tests"]) == 0:
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": bench_data["instance_id"],
                        "shortest_test_type": "RUNNING_FAILED",
                        "shortest_test_failed_before_type":  "RUNNING_FAILED",
                        "f2p_count": 0,
                        "p2f_count": 0,
                        "p2p_count": 0,
                        "f2f_count": 0,
                        "shortest_test": "",
                        "shortest_test_failed_before": "",
                        "tests_info": {
                            PASS_TO_PASS: [],
                            PASS_TO_FAIL: [],
                            FAIL_TO_PASS: [],
                            FAIL_TO_FAIL: [],
                            RUNNING_FAILED: []
                        }
                    }
                )
                + "\n"
            )
        
        
        report["total_instance_count"] += 1
        shortest_test_type = "RUNNING_FAILED"
        shortest_test_failed_before_type = "RUNNING_FAILED"
        
        report["shortest_all"][shortest_test_type + "_count"] += 1
        report["shortest_all"][shortest_test_type].append(instance_id)
        report["shortest_failed_before"][shortest_test_failed_before_type + "_count"] += 1
        report["shortest_failed_before"][shortest_test_failed_before_type].append(instance_id)
        
        with open(f"{args.output_folder}/report.json", "w") as f:
            json.dump(report, f, indent=4)
        return

    final_tests = tests["final_tests"]
    # structure = get_repo_structure(
    #     instance_id, bench_data["repo"], bench_data["base_commit"], "playground"
    # )
    # files, paths, classes = get_full_file_paths_and_classes_and_functions(structure)
    
    instance_image_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key
    
    repo = bench_data["repo"]
    version = bench_data["version"]
    spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
    
    # correct_tests = []
    shortest_test = ""
    shortest_test_type = ""
    shortest_test_failed_before = ""
    shortest_test_failed_before_type = ""
    tests_info = {
        PASS_TO_PASS: [],
        PASS_TO_FAIL: [],
        FAIL_TO_PASS: [],
        FAIL_TO_FAIL: [],
        RUNNING_FAILED: []
    }
    
    # Build + start instance container (instance image should already be built)
    container = build_container(test_spec, client, args.run_id, logger, False, False)
    container.start()
    logger.info(f"Container for {instance_id} started: {container.id}")
    
    run_output_list = []
    test_script = Path(log_dir / "do_test.sh")
    test_script_content, test_command = make_test_script(bench_data, spec, "testbed", "/testbed", bench_data["base_commit"], "", False)
    test_script.write_text(test_script_content)
    logger.info(
        f"Test script for {instance_id} written to {test_script};"
    )
    copy_to_container(container, test_script, Path("/do_test.sh"))
            
    patch_file = Path(log_dir / "gold_patch.diff")
    patch_file.write_text(bench_data['patch'])
    logger.info(
        f"Gold patch for {instance_id} written to {patch_file};"
    )
    logger.info("Applying gold patch to container...")
    copy_to_container(container, patch_file, Path("/tmp/gold_patch.diff"))# Attempt to apply patch to container
    val = container.exec_run(
        "git apply --allow-empty -v /tmp/gold_patch.diff",
        workdir="/testbed",
        user="root",
    )
    if val.exit_code != 0:
        logger.info(f"Failed to apply patch to container, trying again...")
        
        # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
        val = container.exec_run(
            "patch --batch --fuzz=5 -p1 -i /tmp/gold_patch.diff",
            workdir="/testbed",
            user="root",
        )
        if val.exit_code != 0:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}",
                logger,
            )
        else:
            logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
    else:
        logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
            
    try:
        for i, test_dict in enumerate(final_tests):
            test = test_dict['test']
            resolved_before = test_dict['resolved_before']
        
            test_file = Path(log_dir / f"test_{i}.py")
            test_file.write_text(test)
            logger.info(
                f"Intermediate test for {instance_id} written to {test_file}, now applying to container..."
            )
            target_path = MAP_REPO_TO_TEST_PATH.get(repo, MAP_REPO_TO_TEST_PATH["default"])
            copy_to_container(container, test_file, Path(target_path))
            # import pdb; pdb.set_trace()
            # Run eval script, write output to logs
            test_output_after, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test.sh", args.timeout)
            test_output_path_after = log_dir / f"test_output_after_{i}.txt"
            logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
            with open(test_output_path_after, "w") as f:
                f.write(test_output_after)
                logger.info(f"Test output for {instance_id} written to {test_output_path_after}")
                if timed_out:
                    f.write(f"\n\nTimeout error: {args.timeout} seconds exceeded.")
                    logger.info(f"Test timed out after {args.timeout} seconds.")
            
            logger.info(f"Grading answer for {instance_id}...")    

            report_after = get_logs_eval(test_output_path_after)
            resolved_after = set(report_after.values()) == {"PASSED"}
            running_failed = len(report_after) == 0 or timed_out
            logger.info(
                f"After applying patch:\n"
                f"report: {report_after}\n"
                f"Running failed: {running_failed}\n"
                f"Result for {instance_id}: resolved: {resolved_after}"
            )
            
            # test_correct = resolved_after and not resolved_before
            # if test_correct:
            #     correct_tests.append(test)
            
            test_type = map_resolved_to_pass_fail(resolved_before, resolved_after, running_failed)
            test_info = {
                "index": i,
                "type": test_type,
                "test": test,
                "running_failed": running_failed,
                "before": {
                    "resolve": resolved_before,
                },
                "after": {
                    "resolve": resolved_after,
                    "report": report_after,
                    "output": test_output_after,
                }
            }
            if not shortest_test or len(test) < len(shortest_test):
                shortest_test = test
                shortest_test_type = test_type
            if not resolved_before and (not shortest_test_failed_before or len(test) < len(shortest_test_failed_before)):
                shortest_test_failed_before = test
                shortest_test_failed_before_type = test_type
            tests_info[test_type].append(test_info)
            
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
        # shortest
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
        cleanup_container(client, container, logger)
        if False:
            remove_image(client, test_spec.instance_image_key, logger)
    close_logger(logger)
    # exit()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": bench_data["instance_id"],
                    "shortest_test_type": shortest_test_type,
                    "shortest_test_failed_before_type": shortest_test_failed_before_type,
                    "f2p_count": len(tests_info[FAIL_TO_PASS]),
                    "p2f_count": len(tests_info[PASS_TO_FAIL]),
                    "p2p_count": len(tests_info[PASS_TO_PASS]),
                    "f2f_count": len(tests_info[FAIL_TO_FAIL]),
                    "top1_f2p_exist": bool([x for x in tests_info[FAIL_TO_PASS] if x["index"] < 1]),
                    "top2_f2p_exist": bool([x for x in tests_info[FAIL_TO_PASS] if x["index"] < 2]),
                    "top3_f2p_exist": bool([x for x in tests_info[FAIL_TO_PASS] if x["index"] < 3]),
                    "shortest_test": shortest_test,
                    "shortest_test_failed_before": shortest_test_failed_before,
                    "tests_info": tests_info
                }
            )
            + "\n"
        )
    
    if not shortest_test_failed_before:
        shortest_test_failed_before = ""
        shortest_test_failed_before_type = "RUNNING_FAILED"
    
    report["total_instance_count"] += 1
    report["shortest_all"][shortest_test_type + "_count"] += 1
    report["shortest_all"][shortest_test_type].append(instance_id)
    report["shortest_failed_before"][shortest_test_failed_before_type + "_count"] += 1
    report["shortest_failed_before"][shortest_test_failed_before_type].append(instance_id)
    report["exist_f2p_count"] += min(len(tests_info[FAIL_TO_PASS]), 1)
    report["top1_f2p_count"] += bool([x for x in tests_info[FAIL_TO_PASS] if x["index"] < 1])
    report["top2_f2p_count"] += bool([x for x in tests_info[FAIL_TO_PASS] if x["index"] < 2])
    report["top3_f2p_count"] += bool([x for x in tests_info[FAIL_TO_PASS] if x["index"] < 3])
    for test_type in tests_info:
        report["total_tests_all"]["total_tests_count"] += len(tests_info[test_type])
        report["total_tests_all"][test_type + "_count"] += len(tests_info[test_type])
        if test_type.startswith("FAIL"):
            report["total_tests_failed_before"][test_type + "_count"] += len(tests_info[test_type])
            report["total_tests_failed_before"]["total_tests_count"] += len(tests_info[test_type])
    
    with open(f"{args.output_folder}/report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    return

def tests_eval(args):
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    instances = args.target_id
    swe_bench_data = load_dataset(args.dataset, split=args.split)
    if instances is None:
        instances = [x["instance_id"] for x in swe_bench_data]
    tests = load_jsonl(args.tests_file)
    tests = [test for test in tests if test['instance_id'] in instances]
    prev_o = load_jsonl(args.output_file) if args.skip_existing and os.path.exists(args.output_file) else []
    instances_data = [x for x in swe_bench_data if x["instance_id"] in instances]
    temp_instances = {
        instance['instance_id'] : {
            "model_name_or_path": args.model,
            "instance_id": instance['instance_id'],
            "model_patch": "<temp>"
        }
        for instance in instances_data
    }
    dataset = get_dataset_from_preds(args.dataset, args.split, instances, temp_instances, "temp")
    test_specs = list(map(make_test_spec, dataset))
    
    if args.skip_existing:
        for o in prev_o:
            instance_id = o['instance_id']
            shortest_test_type = o["shortest_test_type"]
            shortest_test_failed_before_type = o["shortest_test_failed_before_type"]
            if not shortest_test_failed_before_type:
                shortest_test_failed_before_type = "RUNNING_FAILED"
            tests_info = o["tests_info"]
            report["total_instance_count"] += 1
            report["shortest_all"][shortest_test_type + "_count"] += 1
            report["shortest_all"][shortest_test_type].append(instance_id)
            report["shortest_failed_before"][shortest_test_failed_before_type + "_count"] += 1
            report["shortest_failed_before"][shortest_test_failed_before_type].append(instance_id)
            report["exist_f2p_count"] += min(len(tests_info[FAIL_TO_PASS]), 1)
            for test_type in tests_info:
                report["total_tests_all"]["total_tests_count"] += len(tests_info[test_type])
                report["total_tests_all"][test_type + "_count"] += len(tests_info[test_type])
                if test_type.startswith("FAIL"):
                    report["total_tests_failed_before"][test_type + "_count"] += len(tests_info[test_type])
                    report["total_tests_failed_before"]["total_tests_count"] += len(tests_info[test_type])
        with open(f"{args.output_folder}/report.json", "w") as f:
            json.dump(report, f, indent=4)
        print(f"Skipped {len(prev_o)} existing instances.")
    
    print(f"Running {len(instances)} instances...")
    client = docker.from_env()
    print("Docker client connected.")
    
    
    build_env_images(client, dataset, False, args.num_threads)
    print("Environment images built.")
    
    # print number of existing instance images
    instance_image_ids = {get_instance_image_key(x) for x in swe_bench_data}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not args.force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")


    with open(f"{args.output_folder}/used_tests.jsonl", "w") as f:
        for loc in tests:
            f.write(json.dumps(loc) + "\n")

    results = []

    if args.num_threads == 1:
        for loc in tqdm(tests, total=len(tests)):
            test_spec = [x for x in test_specs if x.instance_id == loc["instance_id"]][0]
            result = instance_tests_eval(loc, test_spec, args, swe_bench_data, prev_o, client)
            if result is not None:
                results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = {
                executor.submit(instance_tests_eval,
                                # print(loc['instance_id'], [x.instance_id for x in test_specs], [x for x in test_specs if x.instance_id == loc["instance_id"]]),
                                loc,
                                [x for x in test_specs if x.instance_id == loc["instance_id"]][0],
                                args, swe_bench_data, prev_o, client,
                ): loc for loc in tests
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(tests)
            ):
                result = future.result()
                if result is not None:
                    results.append(result)
    
    clean_images(client, existing_images, args.cache_level, args.clean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tests_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--loc_interval", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument(
        "--stop_at_n_unique_valid_samples",
        type=int,
        default=-1,
        help="Early stop when we get N unique valid samples, set to -1 if don't want to do early stopping.",
    )
    parser.add_argument("--gen_and_process", action="store_true")
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
    parser.add_argument(
        "--timeout", type=int, default=120, help="Timeout (in seconds) for running tests for each instance"
    )
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-4o", "deepseek-coder", "gpt-4o-mini"],
    )
    parser.add_argument("--target_id", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--only_correct", action="store_true"
    )  # only work on correct loc files (saves time)
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="instance",
    )
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument("--run_id", type=str, default="temp")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    # if not os.path.exists(os.path.join(args.output_folder, "localization_logs")):
    #     os.makedirs(os.path.join(args.output_folder, "localization_logs"))

    args.output_file = os.path.join(args.output_folder, "eval_result.jsonl")

    tests_eval(args)


if __name__ == "__main__":
    main()
