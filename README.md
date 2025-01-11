# README

## 项目成员

|  姓名  |      学号      |
| :----: | :------------: |
| 韩可可 | 22920212204379 |
| 刘昊洋 | 22920212204418 |
| 卢嘉明 | 22920212204176 |

## 项目介绍

本项目是一个基于机器学习和大语言模型（LLM）的代码修复工具，旨在自动化地定位和修复代码中的错误。系统通过多个阶段的处理流程，包括文件级定位、相关代码级定位、编辑位置定位、修复补丁生成、复现测试生成和迭代优化修复，最终生成有效的修复补丁。本工具能够显著提高开发人员在代码修复方面的工作效率。

## 准备

首先，你需要将本项目通过 `git clone` 或者其他方法复制到本地。

然后，请你使用 conda 创建一个虚拟环境并安装依赖。

```bash
conda create -n aglibro python=3.11 
conda activate aglibro
pip install -r requirements.txt
```

接着，请你配置你使用的大语言模型 API Key 和 URL。

```bash
export OPENAI_API_KEY=sk-XXX
# export OPENAI_BASE_URL=https://xxx.xxx.com/v1 # 如果需要使用镜像站，请取消注释
```

最后，请将你存放本项目的路径加入 `PYTHONPATH`。

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 项目阶段及命令

> 提示：
>
> 在第 4 步及之前的步骤中，可以使用 `--target_id <instance_id>` 参数指定执行特定一个任务。
>
> 在第 5 步及之后的步骤中，可以使用 `--target_ids <instance_id>` 参数指定执行特定一个任务。

### 1. 定位到可疑文件

首先，我们定位到可疑文件。这是在一个多步骤过程中完成的，我们将 LLM 定位文件与检索文件组合在一起。

执行以下命令，生成 LLM 预测的可疑文件。

```shell
python aglibro/fl/localize.py --file_level \
                                --output_folder results/file_level \
                                --num_threads 10 \
                                --skip_existing 
```

- **输入**：源代码文件、任务描述。
- **输出**：`results/file_level/loc_outputs.jsonl` 文件，包含 LLM 预测的可疑文件位置。
- **日志**：`results/file_level/localization_logs` 文件夹。

这将把所有 LLM 预测的可疑文件位置保存在 `results/file_level/loc_outputs.jsonl` 文件中，并将日志保存到 `results/file_level/localization_logs` 文件夹。

接下来，我们通过一种简单的基于嵌入的检索方法来补充之前识别的可疑文件，以识别更多的可疑文件。

这一过程首先通过使用 LLM 生成一份不需要检索的无关文件夹列表，从而筛选出无关的文件夹。以下是执行该操作的命令：

```shell
python aglibro/fl/localize.py --file_level \
                                --irrelevant \
                                --output_folder results/file_level_irrelevant \
                                --num_threads 10 \
                                --skip_existing 
```

- **输入**：源代码文件、任务描述。
- **输出**：`results/file_level_irrelevant/loc_outputs.jsonl` 文件，包含识别出的无关文件夹。
- **日志**：`results/file_level_irrelevant/localization_logs` 文件夹。

这将把识别出的无关文件夹保存在 `results/file_level_irrelevant/loc_outputs.jsonl` 文件中，并将日志保存到 `results/file_level_irrelevant/localization_logs` 文件夹。

接下来，我们进行检索（注意，嵌入操作是使用 OpenAI 的 `text-embedding-3-small` 模型完成的），通过传入无关文件夹并运行以下命令：

```shell
python aglibro/fl/retrieve.py --index_type simple \
                                --filter_type given_files \
                                --filter_file results/file_level_irrelevant/loc_outputs.jsonl \
                                --output_folder results/retrievel_embedding \
                                --persist_dir embedding/swe-bench_simple \
                                --num_threads 10 
```

- **输入**：`results/file_level_irrelevant/loc_outputs.jsonl` 文件。
- **输出**：`results/retrievel_embedding/retrieve_locs.jsonl` 文件，包含通过嵌入检索方法得到的文件定位结果。

这将把检索到的文件保存在 `results/retrievel_embedding/retrieve_locs.jsonl` 文件中，并将日志保存到 `results/retrievel_embedding/retrieval_logs` 文件夹。

最后，将 LLM 预测的可疑文件位置与基于嵌入的检索文件合并，以获得最终的相关文件列表：

```shell
python aglibro/fl/combine.py  --retrieval_loc_file results/retrievel_embedding/retrieve_locs.jsonl \
                                --model_loc_file results/file_level/loc_outputs.jsonl \
                                --top_n 3 \
                                --output_folder results/file_level_combined 
```

- **输入**：`results/retrievel_embedding/retrieve_locs.jsonl` 和 `results/file_level/loc_outputs.jsonl` 文件。
- **输出**：`results/file_level_combined/combined_locs.jsonl` 文件，包含合并后的最终定位结果。

`results/file_level_combined/combined_locs.jsonl` 文件包含了最终确定的可疑文件列表。

### 2. 定位到相关元素

接下来，进入定位相关元素的步骤。

运行以下命令，将第一阶段的可疑文件作为输入：

```shell
python aglibro/fl/localize.py --related_level \
                                --output_folder results/related_elements \
                                --top_n 3 \
                                --compress_assign \
                                --compress \
                                --start_file results/file_level_combined/combined_locs.jsonl \
                                --num_threads 10 \
                                --skip_existing 
```

- **输入**：`results/file_level_combined/combined_locs.jsonl` 文件。
- **输出**：`results/related_elements/loc_outputs.jsonl` 文件，包含相关代码元素的定位结果。

这将把相关元素保存在 `results/related_elements/loc_outputs.jsonl` 文件中，并将日志保存到 `results/related_elements/localization_logs` 文件中。

### 3. 定位到编辑位置

最后，使用相关元素，我们将进行编辑位置的定位。这是通过采样来获得多个不同的编辑位置集合:


```shell
python aglibro/fl/localize.py --fine_grain_line_level \
                                --output_folder results/edit_location_samples \
                                --top_n 3 \
                                --compress \
                                --temperature 0.8 \
                                --num_samples 4 \
                                --start_file results/related_elements/loc_outputs.jsonl \
                                --num_threads 10 \
                                --skip_existing 
```

- **输入**：`results/related_elements/loc_outputs.jsonl` 文件。
- **输出**：`results/edit_location_samples/loc_outputs.jsonl` 文件，包含编辑位置的定位结果。

这将把编辑位置保存在 `results/edit_location_samples/loc_outputs.jsonl` 文件中，并将日志保存到 `results/edit_location_samples/localization_logs` 文件夹。

运行以下命令以分离各个编辑位置集：

```shell
python aglibro/fl/localize.py --merge \
                                --output_folder results/edit_location_individual \
                                --top_n 3 \
                                --num_samples 4 \
                                --start_file results/edit_location_samples/loc_outputs.jsonl 
```

- **输入**：`results/edit_location_samples/loc_outputs.jsonl` 文件。
- **输出**：`results/edit_location_individual/loc_merged_${i}-${i}_outputs.jsonl` 文件，包含分离后的编辑位置定位结果。

分离的编辑位置集可以在 `results/edit_location_individual` 中找到。位置文件将命名为 `loc_merged_{x}-{x}_outputs.jsonl`，其中 `x` 表示单独的样本。在我们对 SWE-bench 进行实验时，我们将使用所有 4 个编辑位置集，并分别对它们进行修复，从而生成 4 次不同的修复运行。

### 4. 修复补丁生成

根据编辑位置生成多个修复候选补丁。

```bash
for i in {0..3}; do
    python aglibro/repair/repair.py --loc_file results/edit_location_individual/loc_merged_${i}-${i}_outputs.jsonl \
                                    --output_folder results/repair_sample_$((i+1)) \
                                    --loc_interval \
                                    --top_n=3 \
                                    --context_window=10 \
                                    --max_samples 10  \
                                    --cot \
                                    --diff_format \
                                    --gen_and_process \
                                    --num_threads 2 
done
```

- **输入**：`results/edit_location_individual/loc_merged_${i}-${i}_outputs.jsonl` 文件。
- **输出**：`results/repair_sample_$((i+1))/output.jsonl` 文件，包含生成的修复补丁。

这些命令每个生成 10 个样本（1 个贪婪样本和 9 个通过温度采样生成的样本），由 `--max_samples 10` 参数定义。`--context_window` 表示在提供给模型进行修复的每个本地化编辑位置之前和之后的代码行数。补丁将保存在 `results/repair_sample_{i}/output.jsonl` 中，该文件包含每个样本的原始输出以及任何轨迹信息（例如，标记数量）。完整的日志也会保存在 `results/repair_sample_{i}/repair_logs/` 中。

### 5. 对候选补丁进行排序

```bash
python aglibro/feedback/rerank.py \
    --patch_folder results/repair_sample_1,results/repair_sample_2,results/repair_sample_3,results/repair_sample_4 \
    --num_samples 40 \
    --deduplicate --plausible \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --num_threads 10 \
    --output_folder results/rerank \
    --output_file success_patches.jsonl \
    --skip_existing
```

- 输入文件：`results/repair_sample_$((i+1))/output.jsonl`。包含已经生成的候选补丁。
- 输出文件：`results/rerank/success_patches.jsonl`，包含排序后的候选补丁。

在生成多个修复候选补丁后，系统需要对它们进行排序和筛选，以选择最有可能有效的补丁。这一步通过 `aglibro/feedback/rerank.py` 脚本来完成。该脚本会根据回归测试和复现测试的结果，对候选补丁进行重排序，并去除重复的补丁。排序的依据包括补丁的通过率、测试覆盖率等指标。最终，排序后的补丁列表将存储在 `results/rerank/success_patches.jsonl` 文件中，供后续的迭代优化使用。

### 6. 复现测试生成

生成复现测试用例，验证修复补丁的有效性。

```bash
python aglibro/libro/libro.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --output_folder results/libro \
    --output_file final_tests.jsonl \
    --num_samples 10 \
    --generate_temperature 0.7 \
    --check_temperature 0 \
    --num_threads 10 \
    --skip_existing \
    --max_turns 5 \
    --cost_limit 0.15
```

- **--dataset**：指定使用的数据集。
- **--num_samples**：指定生成的测试用例数量。
- **--max_turns**：指定最大迭代次数。
- **输入**：任务描述和源代码文件。
- **输出**：`results/libro/final_tests.jsonl` 文件，包含生成的复现测试用例。

复现测试生成是验证修复补丁有效性的关键步骤。通过 `aglibro/libro/libro.py` 脚本，系统会生成多个复现测试用例，这些测试用例能够模拟代码中的错误场景，并验证修复补丁是否能够解决问题。生成的测试用例会经过多次迭代优化，以确保其准确性和有效性。最终，生成的复现测试用例将存储在 `results/libro/final_tests.jsonl` 文件中，供后续的迭代优化和补丁验证使用。

### 7. 迭代优化修复

根据复现测试结果对修复补丁进行迭代优化。

```bash
python aglibro/feedback/feedback.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --output_folder results/feedback \
    --output_file all_preds.jsonl \
    --temperature 0 \
    --num_threads 5 \
    --edits_file results/rerank/success_patches.jsonl \
    --tests_file results/libro/final_tests.jsonl \
    --loc_files results/edit_location_individual/loc_merged_0-0_outputs.jsonl,results/edit_location_individual/loc_merged_1-1_outputs.jsonl,results/edit_location_individual/loc_merged_2-2_outputs.jsonl,results/edit_location_individual/loc_merged_3-3_outputs.jsonl \
    --loc_interval --top_n=3 --context_window=10 \
    --cot --diff_format \
    --tests_type libro --tests_top_n 3 --sort_tests \
    --patches_top_n 20 \
    --skip_existing
```

- **--edits_file**：指定修复补丁文件。
- **--tests_file**：指定复现测试文件。
- **--loc_files**：指定使用的定位文件。
- **--model**：指定模型。
- **--patches_top_n**：指定截取排名较高的前多少个补丁。
- **--tests_type**：指定复现测试类型，这里为 Libro 策略生成的测试用例。
- **--tests_top_n**：指定截取前多少个测试用例。
- **--sort_test**：指定需要按照长度排序测试用例。

- **输入**：修复补丁和复现测试用例。
- **输出**：`results/feedback/all_preds.jsonl`，包含最终的修复补丁。

在生成复现测试用例后，系统会根据测试结果对修复补丁进行迭代优化。这一步通过 `aglibro/feedback/feedback.py` 脚本来完成。该脚本会运行复现测试，并根据测试结果对补丁进行优化。如果测试失败，系统会将失败信息和测试内容提供给 LLM，重新生成补丁。这个过程会重复多次，直到生成的补丁能够通过所有测试。最终，优化后的补丁将存储在 `results/feedback/all_preds.jsonl` 文件中。

## 结论

通过以上步骤，本项目能够自动化地定位和修复代码中的错误，生成有效的修复补丁，并通过复现测试验证补丁的有效性。希望本README文件能帮助您更好地理解和使用本项目。