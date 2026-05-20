# 文档主题分类 - DTS406TC 课程作业 1

本仓库包含 DTS406TC 自然语言处理课程作业 1：文档主题分类的实现。

## 概述

本项目实现了完整的文档主题分类流程，包括：

- **数据集收集和预处理**：两个数据集（体育 vs 商业新闻文章）
- **四种分类方法**：朴素贝叶斯、SVM、Word2Vec 和 BERT
- **评估**：所有模型使用一致的评估指标
- **文档**：完整的英文和中文 README

## 项目结构

```
.
├── algorithms/
│   ├── preprocessing/          # 数据集预处理
│   │   ├── preprocess_dataset.py
│   │   ├── split_dataset.py
│   │   ├── dataset_statistics.py
│   │   └── stats_analysis_dataset1.py
│   │
│   ├── traditional/            # 传统机器学习算法
│   │   ├── train_naive_bayes.py
│   │   └── train_svm.py
│   │
│   ├── deep_learning/          # 深度学习算法
│   │   ├── train_word2vec_classifier.py
│   │   └── train_bert_classifier.py
│   │
│   ├── evaluation/             # 评估工具
│   │   ├── evaluate_predictions.py
│   │   ├── aggregate_results.py
│   │   └── plot_results.py
│   │
│   └── utils/                  # 工具函数
│       ├── data_io.py
│       ├── metrics.py
│       ├── text_processing.py
│       └── seed.py
│
├── data/
│   ├── raw/                    # 原始数据集
│   │   ├── dataset_1/          # 数据集 1（如果适用）
│   │   └── dataset_2/          # 数据集 2：AG News
│   ├── processed/              # 预处理后的数据
│   │   ├── dataset_1/          # 数据集 1 预处理数据
│   │   └── dataset_2/          # 数据集 2 预处理数据
│   └── results/                # 模型结果
│       ├── predictions/        # 预测文件
│       ├── metrics/            # 指标文件
│       └── figures/            # 生成的图表
│           └── dataset_2/      # 数据集 2 图表
│
├── dataset_2/                  # 数据集 2 文件
│   ├── raw_data.csv
│   ├── sample_100.csv
│   ├── dataset_info.md
│   ├── label_mapping.csv
│   └── initial_label_distribution.csv
│
├── requirements.txt
├── README.md                   # 英文文档
└── README_cn.md                # 中文文档
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 预处理数据集

```bash
python algorithms/preprocessing/preprocess_dataset.py \
    --dataset_name dataset_2 \
    --input_path dataset_2/raw_data.csv \
    --output_dir data/processed \
    --seed 42
```

### 2. 训练朴素贝叶斯分类器

```bash
python algorithms/traditional/train_naive_bayes.py \
    --dataset_name dataset_2 \
    --train_path data/processed/dataset_2/train.csv \
    --val_path data/processed/dataset_2/val.csv \
    --test_path data/processed/dataset_2/test.csv \
    --output_dir data/results \
    --seed 42
```

### 3. 训练 Word2Vec 分类器

```bash
python algorithms/deep_learning/train_word2vec_classifier.py \
    --dataset_name dataset_2 \
    --train_path data/processed/dataset_2/train.csv \
    --val_path data/processed/dataset_2/val.csv \
    --test_path data/processed/dataset_2/test.csv \
    --output_dir data/results \
    --seed 42 \
    --embedding_dim 100 \
    --window_size 5 \
    --min_count 2 \
    --epochs 10
```

### 4. 评估结果

```bash
python algorithms/evaluation/evaluate_predictions.py \
    --dataset_name dataset_2 \
    --predictions_dir data/results/predictions \
    --metrics_dir data/results/metrics \
    --output_dir data/results/evaluation
```

### 5. 聚合结果

```bash
python algorithms/evaluation/aggregate_results.py \
    --metrics_dir data/results/metrics \
    --output_dir data/results
```

### 6. 生成图表

```bash
python algorithms/evaluation/plot_results.py \
    --dataset_name dataset_2 \
    --metrics_dir data/results/metrics \
    --output_dir data/results/figures
```

## 数据集

### 数据集 2：AG News（体育 vs 商业）

- **路径**：`data/raw/dataset_2/`
- **场景**：新闻主题分类
- **标签**：sports（体育）、business（商业）
- **原始样本数**：60,000
- **划分**：
  - 训练集：42,000
  - 验证集：9,000
  - 测试集：9,000

## 已实现的算法

### 数据集 2

- **朴素贝叶斯**：已实现
- **Word2Vec-based classifier**：已实现（简化版 skip-gram 风格 embedding baseline）

## 依赖项

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

## 作者

- 小组成员 1：[你的名字]
- 小组成员 2：[你的名字]

## 许可证

MIT License
