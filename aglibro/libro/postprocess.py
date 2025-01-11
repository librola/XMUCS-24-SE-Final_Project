import argparse
import concurrent.futures
import json
import os
from difflib import unified_diff
import docker
import traceback
import logging

from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from docker.models.containers import Container

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

POSTPROCESS_LOG_DIR = Path("results/test_postprocess")

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

# def run_generate_test(tests, test_spec : TestSpec, args, swe_bench_data, prev_o, client):
def run_generate_test(
    test_id: int,
    turn_id: int,
    container: docker.models.containers.Container,
    test_spec: TestSpec,
    instance: dict,
    test: str,
    logger: logging.Logger,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    test_command: str
):
    instance_id = instance["instance_id"]
    instance_dir = instance["instance_dir"]

    logger.info(f"================ Running {instance_id} test #{test_id} turn #{turn_id}   ================")
    
    repo = instance["repo"]
    version = instance["version"]
    spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
    
    timeout = 300
    
    # Write test file to instance directory and copy to container
    test_file = Path(instance_dir / f"test_{test_id}_{turn_id}.py")
    test_file.write_text(test)
    logger.info(
        f"Intermediate test for {instance_id} written to {test_file}, now applying to container..."
    )
    target_path = MAP_REPO_TO_TEST_PATH.get(repo, MAP_REPO_TO_TEST_PATH["default"])
    copy_to_container(container, test_file, Path(target_path))
    logger.info(f"Test file copied to container: {target_path}")

    # Run eval script, write output to logs
    test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test.sh", timeout)
    test_output_path = instance_dir / f"test_output_{test_id}_{turn_id}.txt"
    logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
    with open(test_output_path, "w") as f:
        f.write(test_output)
        logger.info(f"Test output for {instance_id} written to {test_output_path}")
        if timed_out:
            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            logger.info(f"Test timed out after {timeout} seconds.")
    
    logger.info(f"Grading answer for {instance_id}...")     
    report_before = get_logs_eval(test_output_path)
    resolved_before = set(report_before.values()) == {"PASSED"}
    logger.info(f"is resolved: {resolved_before}")
    parsed_output = parse_output(test_output, test_command)
    
            
    return resolved_before, parsed_output

def process_generated_tests(tests, test_spec : TestSpec, args, swe_bench_data, prev_o, client):
    instance_id = tests["instance_id"]
    log_dir = POSTPROCESS_LOG_DIR / instance_id
    
    # log_file = os.path.join(
    #     args.output_folder, instance_id, f"{instance_id}.log"
    # )
    log_file = POSTPROCESS_LOG_DIR / instance_id / f"{instance_id}.log"
    logger = setup_logger(instance_id, log_file, mode="a")
    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    if found:
        logger.info(f"skipping {instance_id} since patch already generated")
        return None

    logger.info(f"================ postprocessing {instance_id} ================")
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    
    if len(tests["gen_tests"]) == 0:
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": bench_data["instance_id"],
                        "outputs": []
                    }
                )
                + "\n"
            )

    gen_tests = tests["gen_tests"]
    
    problem_statement = bench_data["problem_statement"]
    structure = get_repo_structure(
        instance_id, bench_data["repo"], bench_data["base_commit"], "playground"
    )
    files, paths, classes = get_full_file_paths_and_classes_and_functions(structure)
    
    instance_image_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key
    
    repo = bench_data["repo"]
    version = bench_data["version"]
    spec = MAP_REPO_VERSION_TO_SPECS[repo][version]
    
    # Run the instance
    container = None
    # Build + start instance container (instance image should already be built)
    container = build_container(test_spec, client, "temp", logger, False, False)
    container.start()
    logger.info(f"Container for {instance_id} started: {container.id}")
    
    run_output_list = []
    test_script = Path(log_dir / "do_test.sh")
    test_script_content, test_command = make_test_script(bench_data, spec, "testbed", "/testbed", bench_data["base_commit"], "", True)
    test_script.write_text(test_script_content)
    copy_to_container(container, test_script, Path("/do_test.sh"))
    logger.info(
        f"Test script for {instance_id} written to {test_script};"
    )
    
    try:
        for i, test in enumerate(gen_tests):
            test_file = Path(log_dir / f"test_{i}.py")
            test_file.write_text(test)
            logger.info(
                f"Intermediate test for {instance_id} written to {test_file}, now applying to container..."
            )
            target_path = MAP_REPO_TO_TEST_PATH.get(repo, MAP_REPO_TO_TEST_PATH["default"])
            copy_to_container(container, test_file, Path(target_path))
            
            # Run eval script, write output to logs
            test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /do_test.sh", args.timeout)
            test_output_path = log_dir / f"test_output_{i}.txt"
            logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
            with open(test_output_path, "w") as f:
                f.write(test_output)
                logger.info(f"Test output for {instance_id} written to {test_output_path}")
                if timed_out:
                    f.write(f"\n\nTimeout error: {args.timeout} seconds exceeded.")
                    logger.info(f"Test timed out after {args.timeout} seconds.")
            
            logger.info(f"Grading answer for {instance_id}...")     
            report_before = get_logs_eval(test_output_path)
            resolved_before = set(report_before.values()) == {"PASSED"}
            logger.info(f"is resolved: {resolved_before}")
            # if not resolved_before:
            run_output_list.append({
                "index": i,
                "test": test,
                "output": parse_output(test_output, test_command),
                "test_command": test_command,
                # "raw_output": test_output,
                "resolved": resolved_before
            })
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
                    "outputs": run_output_list    
                }
            )
            + "\n"
        )
    return

def run_instance(
        test_spec: TestSpec,
        pred: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
    ):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            # link the image build dir in the log dir
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except:
            # some error, idk why
            pass
    log_file = log_dir / "run_instance.log"

    # Set up report file + logger
    report_path = log_dir / "report.json"
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred["model_patch"] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

        # Attempt to apply patch to container
        val = container.exec_run(
            "git apply --allow-empty -v /tmp/patch.diff",
            workdir="/testbed",
            user="root",
        )
        if val.exit_code != 0:
            logger.info(f"Failed to apply patch to container, trying again...")
            
            # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
            val = container.exec_run(
                "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
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

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        copy_to_container(container, eval_file, Path("/eval.sh"))

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        test_output_path = log_dir / "test_output.txt"
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Get git diff after running eval script
        git_diff_output_after = (
            container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        )

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info(f"Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()
    test_specs = list(map(make_test_spec, instances))

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
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


def postprocess(args):
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    instances = args.target_id
    swe_bench_data = load_dataset(args.dataset, split=args.split)
    tests = load_jsonl(args.tests_file)
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
            result = process_generated_tests(loc, test_spec, args, swe_bench_data, prev_o, client)
            if result is not None:
                results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = {
                executor.submit(process_generated_tests,
                                loc,
                                [x for x in test_specs if x.instance_id == loc["instance_id"]][0],
                                args, swe_bench_data, prev_o, client
                ): loc for loc in tests
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(tests)
            ):
                result = future.result()
                if result is not None:
                    results.append(result)
    
    clean_images(client, existing_images, args.cache_level, args.clean)


def post_process_raw_output(
    raw_output_text, file_contents, logger, file_loc_intervals, args
):
    git_diffs = ""
    raw_git_diffs = ""
    lint_success = False
    content = ""
    try:
        edited_file, new_content = _post_process_multifile_repair(
            raw_output_text,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=args.diff_format,
        )
        if edited_file in file_contents:
            content = file_contents[edited_file]

            git_diff = fake_git_repo("playground", edited_file, content, new_content)

            raw_git_diffs += "\n" + git_diff.replace(
                "\ No newline at end of file\n", ""
            )

            syntax_success = check_syntax(new_content)
            lint_success, prev_errors, errors = lint_code(
                "playground", "test.py", new_content, file_contents[edited_file]
            )

            differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                new_content, file_contents[edited_file]
            )

            print(lint_success, prev_errors, errors, differ_by_empty_lines)

            if syntax_success and not differ_by_empty_lines:
                git_diffs = raw_git_diffs
            else:
                git_diffs = ""  # no need to evaluate
        else:
            diff = list(
                unified_diff(
                    content.split("\n"),
                    new_content.split("\n"),
                    fromfile=edited_file,
                    tofile=edited_file,
                    lineterm="",
                )
            )
            print("Failed parsing diff!")
            print("\n".join(diff))
    except Exception as e:
        print(raw_output_text)
        print(e)

    return git_diffs, raw_git_diffs, content


def post_process_repair(args):
    """
    apply some diff formatting.
    """
    raw_outputs = load_jsonl(args.raw_output_file)
    locs = load_jsonl(args.tests_file)

    for raw_output in raw_outputs:
        instance_id = raw_output["instance_id"]
        log_file = os.path.join(
            args.output_folder, "localization_logs", f"{instance_id}.log"
        )
        logger = setup_logger(log_file)

        if raw_output["raw_output"] == "":
            with open(args.output_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "model_name_or_path": "aglibro",
                            "instance_id": instance_id,
                            "model_patch": "",
                        }
                    )
                    + "\n"
                )
            continue

        if args.select_id == -1:
            # Use the last generation
            assert False, "not implemented for now"
        else:
            # Use the indexed generation
            generation_idx = args.select_id
            print("got into process postprocess")
            try:
                raw_output_text = raw_output["all_generations"][0][generation_idx]
                original_file_content = raw_output["prev_content"][0][generation_idx]
                pred_file = raw_output["file_names"][0][generation_idx]

                pred_files = [loc for loc in locs if loc["instance_id"] == instance_id][
                    0
                ]["found_files"][: args.top_n]

                git_diffs = ""
                raw_git_diffs = ""
                if isinstance(raw_output["raw_output"], str):
                    # for backward compatibility
                    raw_output["raw_output"] = [raw_output["raw_output"]]

                file_contents = {pred_file: original_file_content}

                file_loc_intervals = dict()

                loc = [loc for loc in locs if loc["instance_id"] == instance_id][0]

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

        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "aglibro",
                        "instance_id": instance_id,
                        "model_patch": git_diffs.lstrip(),
                        "raw_model_patch": raw_git_diffs.lstrip(),
                        "original_file_content": content,
                    }
                )
                + "\n"
            )


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
        "--timeout", type=int, default=300, help="Timeout (in seconds) for running tests for each instance"
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

    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, "localization_logs")):
        os.makedirs(os.path.join(args.output_folder, "localization_logs"))

    args.output_file = os.path.join(args.output_folder, "output.jsonl")


    postprocess(args)


if __name__ == "__main__":
    main()
