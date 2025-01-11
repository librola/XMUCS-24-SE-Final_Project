from pathlib import Path
import json
from aglibro.util.utils import load_jsonl

feedback_dir = Path('results/feedback-oracle/instance_logs')

map_success_to_turns = {}

for instance_dir in feedback_dir.iterdir():
    instance_id = instance_dir.name
    
    min_id = 1000000
    for patch_file in instance_dir.glob(f'{instance_id}_*.json'):
        patch_id = int(patch_file.stem[len(instance_id) + 1:])
        with patch_file.open() as f:
            report = json.load(f)
        assert 'final_libro_test_pass_num' not in report or (report['status'] == 'success') == (report['final_libro_test_pass_num'] == 1), report['status']
        if report['status'] == "success" and patch_id < min_id:
            min_id = patch_id
            min_report = report
    if min_id == 1000000:
        continue
    report = min_report
            
    num_turns = len(report['log']) // 2 - 1
    map_success_to_turns[instance_id] = num_turns

print(map_success_to_turns)