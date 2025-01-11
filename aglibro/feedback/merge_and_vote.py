from aglibro.util.postprocess_data import extract_python_blocks, normalize_patch
from aglibro.util.utils import load_jsonl
import json

generated_patch_1 = "results/feedback-1/generated_patches.jsonl"
generated_patch_2 = "results/feedback-2/generated_patches.jsonl"

generated_patch_1 = load_jsonl(generated_patch_1)
generated_patch_2 = load_jsonl(generated_patch_2)

generated_patch_1 = {patch["instance_id"]: patch for patch in generated_patch_1}
generated_patch_2 = {patch["instance_id"]: patch for patch in generated_patch_2}

for instance_id in generated_patch_1:
    if instance_id in generated_patch_2:
        iter2 = generated_patch_2[instance_id]['all_patches']
        iter1 = generated_patch_1[instance_id]['all_patches']
        
        for i in iter2:
            iter1[i].extend(iter2[i])

for instance_id in generated_patch_2:
    if instance_id not in generated_patch_1:
        generated_patch_1[instance_id] = generated_patch_2[instance_id]

for instance in generated_patch_1.values():
    instance_id = instance['instance_id']
    choose_result = []
    choose_id = -1
    iter_result = instance['all_patches']
    for i in range(len(iter_result) - 2, -2, -1):
        i = str(i)
        if iter_result[i]:
            choose_result = iter_result[i]
            choose_id = i
            break

    patch_cnt = {"": 0}
    for patch in choose_result:
        if not patch['patch'].strip():
            continue
        # normalized_patch = normalize_patch(instance_id, patch['patch'], patch['trans_res']['original_file_content'])
        normalized_patch = patch['patch']

        patch_cnt[normalized_patch] = patch_cnt.get(normalized_patch, 0) + 1

    assert patch_cnt[""] == 0
    majority = ""
    for patch in patch_cnt:
        if patch_cnt[patch] > patch_cnt[majority]:
            majority = patch
        
    with open("results/all_preds.jsonl", "a") as f:
        f.write(json.dumps({
                "model_name_or_path": "aglibro",
                "instance_id": instance_id,
                "patch_type": choose_id,
                "model_patch": majority,
        }) + "\n")