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

| 策略 | 适用场景 | 关键参数 |
|------|----------|----------|
| `topic_driven` | 明确领域覆盖需求，按主题/难度/题型分配 | `topics` + `total_count` |
| `seed_driven` | 有少量高质量种子数据，扩增同风格样本 | `seed_file` + `example_num` + `target_count` |
| `self_instruct` | 自主探索多样性，无需外部数据 | `target_count` |

详细参数说明见 [docs/config.md](docs/config.md#生成策略)。

## 核心功能

### 数据生成

- 三种策略可独立或组合使用，按 `weight` 比例分配生成量
- **多角度正交生成**：可配置任意多个维度自动正交组合，Jinja2 按比例均分，代码零改动
- 支持单轮（instruction/output）和多轮对话

数据生成链路：`topics + knowledge`（手动指定）→ Planner LLM 生成 `sub_topic + angle` → Executor LLM 生成 `instruction + output`。每条输出 `metadata` 携带完整维度标签。

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

## 项目结构

```
alembic/
├── api/                   # API 适配层（OpenAI 兼容 + 重试）
│   ├── base.py            #   BaseAPIClient, RetryConfig, RetryWrapper
│   ├── factory.py         #   create_client() 工厂函数
│   ├── providers.py       #   OpenAICompatibleClient
│   └── embedding.py       #   EmbeddingClient（独立去重 API）
├── cleaner/               # 数据清洗
│   ├── cleaner.py         #   DatasetCleaner（文本清洗 + 去重）
│   └── ops.py             #   低层清洗函数
├── core/                  # 核心管线
│   ├── pipeline.py        #   生成 → 清洗 → 评分 编排
│   ├── observer.py        #   Observer 模式（日志 + 统计）
│   ├── stats.py           #   StatisticsCollector
│   ├── inspector.py       #   DatasetInspector（view 命令）
│   └── types.py           #   数据类型定义
├── prompts/               # 提示词系统
│   ├── builder.py         #   PromptBuilder（Jinja2 渲染）
│   └── templates/         #   22 个 .j2 模板（en/zh × 单轮/多轮）
├── quality/               # 质量校验（Chain of Responsibility）
│   └── validators.py      #   Length → Truncation → Dedup
├── scoring/               # LLM 打分
│   └── scorer.py          #   DatasetScorer（多维度并发打分）
├── strategies/            # 生成策略（Strategy 模式）
│   ├── base.py            #   GenerationStrategy 抽象基类
│   ├── composite.py       #   策略编排 + 工厂函数
│   ├── topic_driven.py    #   主题驱动
│   ├── seed_driven.py     #   种子驱动
│   └── self_instruct.py   #   自我指令
├── writers/               # 数据输出
│   └── jsonl_writer.py    #   JSONLWriter（alpaca/chatml/sharegpt）
├── config.py              # 配置解析（6 个 dataclass）
└── cli.py                 # CLI 入口（Click）
tests/                     # pytest 测试（63 项）
```

## 开发

```bash
# 依赖
pip install openai click pyyaml jinja2 tqdm numpy pytest ruff

# 测试
pytest tests/ -v

# Lint
python -m ruff check .
```
