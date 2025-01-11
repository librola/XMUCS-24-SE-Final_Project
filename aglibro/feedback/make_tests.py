import json
from aglibro.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    write_jsonl
    # setup_logger,
)
from pathlib import Path

final_tests = Path("results/libro/final_tests.jsonl")
final_tests = load_jsonl(final_tests)
output_folder = Path("results/feedback-oracle")

for instance in final_tests:
    instance_id = instance["instance_id"]
    tests = instance["final_tests"]
    tests.sort(key=lambda x: len(x["test"]))
    tests = [x['test'] for x in tests]
    instance["final_tests"] = tests
    
if output_folder.exists() is False:
    output_folder.mkdir(parents=True)
write_jsonl(final_tests, output_folder / "tests.jsonl")


import json
from aglibro.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    write_jsonl
    # setup_logger,
)
# from pathlib import Path

# final_tests = Path("results/tests_group/final_tests.jsonl")
# final_tests = load_jsonl(final_tests)

# for instance in final_tests:
#     instance_id = instance["instance_id"]
#     tests = instance["final_tests"]
#     tests = [x['test'] for x in tests]
#     instance["final_tests"] = tests[:2]
    
# if Path('results/feedback').exists() is False:
#     Path('results/feedback').mkdir(parents=True)
# write_jsonl(final_tests, "results/feedback/tests.jsonl")

# eval_result_file = Path('results/tests_group_eval/eval_result.jsonl')
# eval_results = load_jsonl(eval_result_file)
# top1_f2p = []
# for instance in eval_results:
#     instance_id = instance["instance_id"]
#     if 'top1_f2p_exist' in instance and instance['top1_f2p_exist']:
#         top1_f2p.append(instance_id)

# print(" ".join(top1_f2p))