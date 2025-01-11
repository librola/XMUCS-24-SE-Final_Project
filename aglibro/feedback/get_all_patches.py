from pathlib import Path
import json
from aglibro.util.utils import load_jsonl

feedback_dir = Path('results/feedback-pre-ranked/instance_logs')
output_folder = Path('results/feedback-pre-ranked/results-new')
rerank_patches_file = Path('results/rerank/success_patches.jsonl')

if not output_folder.exists():
    output_folder.mkdir(parents=True)

for i in range(10):
    with open(output_folder / (f"all_preds-new-{i}.jsonl" if i else "all_preds-new.jsonl"), "w") as f:
        pass

all_num = 0
stop_num = 0
processed_ids = set()
for instance_dir in feedback_dir.iterdir():
    instance_id = instance_dir.name
    
    iter_result = { -1: [], 0: [], 1: [] }
    # 枚举目录下所有诸如 instance_id + '_' + 数字 + '.json' 的文件
    for patch_file in instance_dir.glob(f'{instance_id}_*.json'):
        patch_id = int(patch_file.stem[len(instance_id) + 1:])
        with patch_file.open() as f:
            report = json.load(f)
        
        if 'final_libro_test_pass_num' not in report:
            print(f"Error: {patch_file}")
            exit(0)
        
        if report['final_libro_test_pass_num'] == 0:
            cur_patch = report['log'][0]['patch']
            report['final_libro_test_pass_num'] = 0
            report['final_result'] = report['log'][1]['result']
            report['final_status'] = report['final_result']['success']
        else:
            assert report['status'] == 'success' and report['final_status'] == 'PASSED'
            cur_patch = report['patch']
        if cur_patch:
            iter_result[report['final_libro_test_pass_num']].append({
                "id": patch_id,
                "patch": cur_patch,
                "final_libro_test_pass_num": report['final_libro_test_pass_num'],
                "final_status": report['final_status'],
                "final_result": report['final_result'],
            })
        else:
            iter_result[-1].append({
                "id": patch_id,
                "patch": cur_patch,
                "final_libro_test_pass_num": -1,
                "final_status": '',
                "final_result": {},
            })
    
    if not iter_result[1] and not iter_result[0] and not iter_result[-1]:
        continue
    
    top10 = []
    iter_result[1].sort(key=lambda x: x['id'])
    iter_result[0].sort(key=lambda x: x['id'])
    for i in [1, 0]:
        for res in iter_result[i]:
            top10.append({
                "patch": res['patch'],
                "patch_type": i
            })
            if len(top10) >= 10:
                break
        if len(top10) >= 10:
            break
    
    while len(top10) < 10:
        top10.append({
            "patch": "",
            "patch_type": -1
        })
    
    for i in range(10):
        with open(output_folder / (f"all_preds-new-{i}.jsonl" if i else "all_preds-new.jsonl"), "a") as f:
            f.write(json.dumps({
                "model_name_or_path": "aglibro",
                "instance_id": instance_id,
                "patch_type": top10[i]['patch_type'],
                "model_patch": top10[i]['patch'],
            }) + "\n")
    processed_ids.add(instance_id)

rerank_patches = load_jsonl(rerank_patches_file)

for rerank_patch in rerank_patches:
    if rerank_patch['instance_id'] in processed_ids:
        continue
    for i in range(10):
        with open(output_folder / (f"all_preds-new-{i}.jsonl" if i else "all_preds-new.jsonl"), "a") as f:
            if i < len(rerank_patch['edits']):
                f.write(json.dumps({
                    "model_name_or_path": "aglibro",
                    "instance_id": rerank_patch['instance_id'],
                    "patch_type": 0,
                    "model_patch": rerank_patch['edits'][i]['model_patch'],
                }) + "\n")
            else:
                f.write(json.dumps({
                    "model_name_or_path": "aglibro",
                    "instance_id": rerank_patch['instance_id'],
                    "patch_type": -1,
                    "model_patch": "",
                }) + "\n")