# DTS406TC Coursework 1：文档主题分类

## 1. 项目简介

本仓库用于 DTS406TC Natural Language Processing Coursework 1。主题是文档主题分类（Document Topic Classification）。项目目标是搭建一个可复现的 Python 流水线，用于数据集收集、文本预处理、主题分类模型训练、预测评估以及生成报告可用的结果材料。

当前阶段：初始框架搭建。仓库已经包含项目目录、数据检查工具、预处理工具、评估工具，以及四个算法的统一脚本入口。四个模型的正式训练逻辑尚未实现。

## 2. 作业要求概述

最终课程项目必须支持：

- 两个文档主题分类数据集；
- 每个数据集至少 3000 条样本；
- 两个数据集应代表不同分类场景；
- 四种算法都要应用到两个数据集：
  - Naive Bayes；
  - SVM；
  - Word2Vec-based classifier；
  - BERT-based classifier；
- 至少使用 precision、recall、F1-score 进行评估；
- 使用 CSV 保存预测结果、指标、汇总表和图表所需数据；
- 支持最终 group report 和 individual literature review 的材料整理。

所有实现代码必须使用 Python。

## 3. 当前项目状态

本初始化阶段已完成：

- 标准项目目录结构；
- 英文 README 和中文 README；
- 原始数据集交付格式检查脚本；
- 通用数据读取、随机种子、文本处理和指标工具；
- 基础预处理、数据划分和统计脚本；
- 四个模型脚本的统一接口模板；
- 预测评估、指标汇总和结果绘图脚本；
- 轻量级 requirements 文件。

尚未实现：

- Naive Bayes 的实际训练逻辑；
- SVM 的实际训练逻辑；
- Word2Vec 分类器的实际训练逻辑；
- BERT 分类器训练或特征提取逻辑；
- 真实数据集和真实实验结果。

当前项目骨架不会生成虚假数据集、虚假预测或虚假 metrics。

## 4. 推荐目录结构

```text
.
├── AGENTS.md
├── README.md
├── README_cn.md
├── requirements.txt
├── docs/
├── algorithms/
│   ├── preprocessing/
│   │   ├── validate_raw_dataset.py
│   │   ├── preprocess_dataset.py
│   │   ├── split_dataset.py
│   │   └── dataset_statistics.py
│   ├── traditional/
│   │   ├── train_naive_bayes.py
│   │   └── train_svm.py
│   ├── deep_learning/
│   │   ├── train_word2vec_classifier.py
│   │   └── train_bert_classifier.py
│   ├── evaluation/
│   │   ├── evaluate_predictions.py
│   │   ├── aggregate_results.py
│   │   └── plot_results.py
│   └── utils/
│       ├── data_io.py
│       ├── metrics.py
│       ├── text_processing.py
│       └── seed.py
├── data/
│   ├── raw/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   ├── processed/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   └── results/
│       ├── predictions/
│       ├── metrics/
│       ├── tables/
│       └── figures/
├── report/
│   ├── group_report/
│   └── individual_literature_reviews/
└── notes/
```

## 5. 数据集收集格式

每个原始数据集应作为一个文件夹提交到 `data/raw/` 下，例如 `data/raw/dataset_1/`。

必需文件：

```text
raw_data.csv
sample_100.csv
dataset_info.md
label_mapping.csv
initial_label_distribution.csv
```

`raw_data.csv` 至少必须包含：

```csv
id,text,label
```

规则：

- `id` 不能为空且必须唯一；
- `text` 不能为空；
- `label` 不能为空；
- 每个数据集至少 3000 行；
- `sample_100.csv` 必须正好 100 行，并保留相同必需列；
- `label_mapping.csv` 必须包含 `original_label,unified_label,description`；
- `initial_label_distribution.csv` 必须包含 `label,count,percentage`。

## 6. 原始数据检查命令

检查一个已提交的原始数据集文件夹：

```bash
python algorithms/preprocessing/validate_raw_dataset.py \
  --dataset_dir data/raw/dataset_1
```

脚本会在终端输出清晰检查结果，并将 validation report 写入 `notes/`。如果发现严重错误，脚本会返回非零退出码。

## 7. 处理后数据格式

预处理和划分完成后，每个处理后数据集应包含：

```text
data/processed/dataset_1/
├── cleaned.csv
├── train.csv
├── val.csv
├── test.csv
├── statistics.csv
├── label_distribution.csv
└── word_frequency.csv
```

划分文件的最低必需列为：

```csv
id,text,label
```

推荐划分比例为 70% train、15% validation、15% test，并使用固定随机种子，例如 42。

## 8. 预处理、划分和统计流程

清洗原始数据集：

```bash
python algorithms/preprocessing/preprocess_dataset.py \
  --input_path data/raw/dataset_1/raw_data.csv \
  --output_path data/processed/dataset_1/cleaned.csv
```

创建 train/validation/test 划分：

```bash
python algorithms/preprocessing/split_dataset.py \
  --input_path data/processed/dataset_1/cleaned.csv \
  --dataset_name dataset_1 \
  --output_dir data/processed/dataset_1 \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

对一个处理后的 CSV 生成统计信息，通常使用 `train.csv`：

```bash
python algorithms/preprocessing/dataset_statistics.py \
  --input_path data/processed/dataset_1/train.csv \
  --dataset_name dataset_1 \
  --output_dir data/processed/dataset_1
```

统计脚本会输出：

- `statistics.csv`；
- `label_distribution.csv`；
- `word_frequency.csv`。

## 9. 算法脚本接口

所有模型脚本使用相同命令行接口：

```bash
python algorithms/traditional/train_naive_bayes.py \
  --dataset_name dataset_1 \
  --train_path data/processed/dataset_1/train.csv \
  --val_path data/processed/dataset_1/val.csv \
  --test_path data/processed/dataset_1/test.csv \
  --output_dir data/results \
  --seed 42
```

当前可用的接口模板脚本：

- `algorithms/traditional/train_naive_bayes.py`
- `algorithms/traditional/train_svm.py`
- `algorithms/deep_learning/train_word2vec_classifier.py`
- `algorithms/deep_learning/train_bert_classifier.py`

这些脚本只用于验证统一接口，并会在路径检查后抛出 `NotImplementedError`。它们不会训练模型，也不会保存虚假输出。

## 10. 结果 CSV 格式

后续模型实现必须将预测文件保存到：

```text
data/results/predictions/{dataset_name}_{model_name}_predictions.csv
```

预测文件必需列：

```csv
id,text,true_label,predicted_label
```

后续模型实现必须将指标文件保存到：

```text
data/results/metrics/{dataset_name}_{model_name}_metrics.csv
```

指标文件必需列：

```csv
dataset,model,feature_type,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,accuracy,train_time_sec,inference_time_sec,random_seed
```

## 11. 评估和汇总流程

评估一个已有 prediction CSV：

```bash
python algorithms/evaluation/evaluate_predictions.py \
  --prediction_path data/results/predictions/dataset_1_naive_bayes_predictions.csv \
  --dataset_name dataset_1 \
  --model_name naive_bayes \
  --feature_type tfidf \
  --output_path data/results/metrics/dataset_1_naive_bayes_metrics.csv \
  --seed 42
```

汇总所有 metrics CSV：

```bash
python algorithms/evaluation/aggregate_results.py \
  --metrics_dir data/results/metrics \
  --output_dir data/results/tables
```

根据汇总表绘制 macro F1 对比图：

```bash
python algorithms/evaluation/plot_results.py \
  --summary_path data/results/tables/all_metrics_summary.csv \
  --output_dir data/results/figures
```

绘图脚本只会基于已经存在的真实 summary CSV 生成图表。

## 12. 当前已实现的内容

以下模型脚本已完整实现：

- SVM：基于 TF-IDF + LinearSVC 的分类模型已实现（由组员 A 实现）；
- BERT-based classifier：基于 DistilBERT 序列分类微调的模型已实现（由组员 B 实现）。

其余模型脚本仍为模板：
- Naive Bayes：interface only，由组员 A 后续实现；
- Word2Vec-based classifier：interface only，由组员 B 后续实现。

## 13. 后续组员如何接入算法实现

组员实现模型时应：

1. 保持现有命令行参数不变；
2. 从给定路径读取 `train.csv`、`val.csv` 和 `test.csv`；
3. 只在训练集上拟合 vectorizer、embedding 和模型；
4. validation 数据只用于调参或 early stopping；
5. 最终性能只在 test set 上评估；
6. 按标准路径和列名保存 prediction 和 metrics CSV；
7. 如果依赖、命令或输出格式发生变化，同步更新 `README.md` 和 `README_cn.md`；
8. 避免写死本机绝对路径，不写没有 CSV 结果支持的报告结论。

## 14. 依赖

项目依赖列在 `requirements.txt`：

```text
pandas
numpy
scikit-learn
matplotlib
torch
transformers
```

安装命令：

```bash
pip install -r requirements.txt
```
