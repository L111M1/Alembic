# Alembic

基于 LLM API 的轻量级 SFT 训练数据生成、清洗和评分管线。命名源自蒸馏器（alembic），寓意将原始数据蒸馏提纯。

## 安装

```bash
pip install openai click pyyaml jinja2 tqdm numpy
```

## 快速开始

```bash
# 1. 设置 API 密钥
export API_KEY=sk-xxx
export BASE_URL=https://your-endpoint/v1

# 2. 生成 + 清洗（一键）
python -m alembic.cli generate --config sft_gen_config.yaml

# 3. 查看数据分布
python -m alembic.cli view generated_sft.jsonl
```

## CLI 命令

| 命令 | 用途 |
|------|------|
| `generate --config CONFIG` | 生成 SFT 数据（含自动清洗） |
| `clean INPUT [-o OUTPUT] [--config CONFIG]` | 独立清洗 JSONL 数据集 |
| `score INPUT [-o OUTPUT] [--config CONFIG]` | LLM 多维度质量打分 |
| `view INPUT [-n N] [-j]` | 查看数据统计、分布、样本 |
| `list-templates` | 列出所有提示词模板 |

常用选项：

```bash
# 预览模式（不写文件）
python -m alembic.cli generate --config config.yaml --dry-run --count 5

# 清洗已有数据
python -m alembic.cli clean input.jsonl -o output.jsonl --config config.yaml

# 查看数据分布，显示 10 条样本
python -m alembic.cli view output.jsonl -n 10

# JSON 格式输出统计报告
python -m alembic.cli view output.jsonl -j
```

## 生成策略

| 策略 | 变体 | 适用场景 | 关键参数 |
|------|------|----------|----------|
| `topic_driven` | 主题驱动 | 按主题+难度+题型生成 | `topics` + `total_count` |
| `seed_driven` | 种子扩增 | 少量种子扩增同风格 | `seed_file` + `target_count` |
| `seed_driven` + `evolution` | 遗传进化 | 种子 + 交叉/变异算子 | `evolution.crossover_rate` + `mutation_types` |
| `evol_instruct` | 迭代进化 | 种子逐轮深度/广度变异 | `seed_file` + `max_rounds` |

- **`topic_driven`**：指定主题和维度，LLM 规划子主题和角度后分批生成
- **`seed_driven`**：基于种子 few-shot 学习，可选遗传算子（交叉/变异）增强多样性
- **`evol_instruct`**：WizardLM 式迭代进化，N 轮深度（加约束/加深/具体化/推理链）和广度变异使指令逐步复杂，再统一生成回答
- **`CompositeStrategy`**：多策略按 `weight` 加权组合，`merge_generators` 交错输出

详细参数说明见 [docs/config.md](docs/config.md#策略编排)。

## 核心功能

### 数据生成

三种策略可独立或按 `weight` 比例组合使用：

- **`topic_driven`**：`topics + knowledge` → Planner LLM 规划 `sub_topic + angle` → Executor LLM 生成 `instruction + output`。支持任意多维度正交组合
- **`seed_driven`**：按概率走交叉（合并两个种子）/ 变异（自定义算子）/ 默认 few-shot 三种模式
- **`evol_instruct`**：两阶段——种子经 N 轮深度+广度迭代进化，再统一生成回答。元数据携带完整进化链

支持单轮（instruction/output）和多轮对话。每条输出 `metadata` 携带策略、话题、维度标签。

### 数据清洗

```
生成阶段 → raw.jsonl → 清洗 → raw_cleaned.jsonl
```

- **文本清洗**：去除 HTML 标签、URL、邮箱、特殊字符，词汇/字符重复过滤
- **长度过滤**：可配置 instruction / output 的 min/max 长度
- **文本指纹去重**：SHA256 精确匹配
- **语义去重**：embedding 向量余弦相似度去重（独立 API，可选）

### 质量打分

LLM-as-Judge 多维度打分，维度完全可自定义：

```yaml
scoring:
  model: gpt-4o
  lang: zh
  dimensions:
    - name: correctness
      label: "准确性"
      max_score: 10
    - name: helpfulness
      label: "实用性"
      max_score: 10
```

输出每条数据附加 `scores` 和 `total_score`，可按阈值过滤低分样本。

### 数据查看

`view` 命令提供数据集的完整统计画像：

- 总量、单轮/多轮分布
- 指令/输出长度分布（min/max/mean/median/p25/p75/p90）
- 主题分布、策略分布
- 评分分布（如有 scores 字段）
- 样本预览（截断 120 字符）

## 配置

完整配置参考 [docs/config.md](docs/config.md)。最小配置示例：

```yaml
api:
  model: qwen-plus
  lang: zh
  concurrency: 4

strategies:
  - type: topic_driven
    dimensions:
      - name: difficulty
        vals: [入门, 进阶, 高级]
      - name: cognitive_level
        vals: [记忆, 理解, 应用, 分析, 评价, 创造]
      - name: question_type
        vals: [问答题, 选择题, 判断题, 填空题]
    topics:
      - topic: "Python 编程"
        weight: 1
        knowledge: "Python 语法、数据类型、控制流、函数、面向对象"
    total_count: 100

output:
  path: ./generated_sft.jsonl
  format: alpaca
```

环境变量：`API_KEY` / `BASE_URL`（Chat API），`EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`（语义去重嵌入 API）。

## 开发

```bash
# 依赖
pip install openai click pyyaml jinja2 tqdm numpy pytest ruff

# 测试
pytest tests/ -v

# Lint
python -m ruff check .
```
