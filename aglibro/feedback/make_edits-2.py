from aglibro.util.utils import load_jsonl, write_jsonl
import json
from pathlib import Path

def process(dir):
    dir = Path(dir)
    
    origin = load_jsonl(dir / "output.jsonl")
    map_id_to_prompt = {
        patch["instance_id"] : patch["traj"]
        for patch in origin
    }
    map_id_to_raw_outputs = {
        patch["instance_id"] : patch['raw_output']
        for patch in origin
    }
    
    final_edits = {
        x : {
            "instance_id": x,
            "edits": []
        }
        for x in map_id_to_prompt.keys()
    }
    
    for i in range(21):
        file = f"output_{i}_normalized.jsonl"
        patches = load_jsonl(dir / file)
        
        for patch in patches:
            instance_id = patch["instance_id"]

            raw = map_id_to_raw_outputs[instance_id][i]
            prompt = map_id_to_prompt[instance_id][i]
            query_result = raw
            if '```python\n' in query_result:
                query_result = query_result.split('```python\n')[1].split('```')[0]
                stop = False
            else:
                query_result = ""
                stop = True
            final_edits[instance_id]["edits"].append({
                "model_patch": patch['model_patch'],
                "raw_edit": query_result,
                "prompt": prompt["prompt"],
                "raw_model_patch": patch['raw_model_patch'],
                "original_file_content": patch['original_file_content']
            })
    return final_edits

result = process("results/repair_run_2/")
# result2 = process("results/repair_run_2/")

# for instance_id, data in result1.items():
#     if instance_id in result2:
#         data["edits"].extend(result2[instance_id]["edits"])
# for instance_id, data in result2.items():
#     if instance_id not in result1:
#         result1[instance_id] = data

result = list(result.values())
write_jsonl(result, "results/feedback/edits_2.jsonl")
