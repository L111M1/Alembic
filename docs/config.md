# 配置参考

Alembic 通过 YAML 文件控制生成、清洗、评分的全部行为。使用方式见 [README.md](../README.md#cli-命令)。

## 完整示例

```yaml
api:
  model: deepseek-v4-flash
  api_key: ""              # 或通过环境变量 API_KEY 设置
  base_url: ""             # 或通过环境变量 BASE_URL 设置
  lang: zh
  concurrency: 10
  params:
    temperature: 0.8
    max_tokens: 2048
  retry:
    max_retries: 3

strategies:
  - type: topic_driven
    weight: 0.5
    topics:
      - topic: "Python 编程基础"
        weight: 3
        knowledge: "Python 是动态类型语言，支持面向对象、函数式编程。"
      - topic: "机器学习"
        weight: 2
        knowledge: "三大范式：监督学习、无监督学习、强化学习。"
      - topic: "数据库与 SQL"
        weight: 1
        knowledge: "关系型：MySQL、PostgreSQL，ACID 事务。"
    total_count: 100

  - type: seed_driven
    weight: 0.3
    seed_file: ./seeds.jsonl
    example_num: 2
    target_count: 30

  - type: self_instruct
    weight: 0.2
    target_count: 20

quality:
  instruction_min_len: 5
  instruction_max_len: 2000
  output_min_len: 30
  output_max_len: 6000
  dedup: true
  remove_truncated: true

cleaner:
  remove_html: true
  remove_urls: true
  remove_emails: true
  max_special_char_ratio: 0.3
  max_word_repetition_ratio: 0.5
  dedup: true
  embedding_dedup: false
  embedding_model: "text-embedding-3-small"
  embedding_similarity_threshold: 0.85

output:
  path: ./generated_sft.jsonl
  format: alpaca
  checkpoint: false
```

## API 配置

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 是 | — | LLM 模型名称 |
| `api_key` | string | 否 | `$API_KEY` | 建议通过环境变量设置 |
| `base_url` | string | 否 | `$BASE_URL` | 建议通过环境变量设置 |
| `lang` | string | 否 | `en` | 生成/提示语言（`zh` / `en`） |
| `concurrency` | int | 否 | — | 并行 API 调用数（TopicDriven / SeedDriven 生效） |
| `params` | dict | 否 | `{}` | LLM 调用参数：`temperature`（建议 0.8~0.95）、`max_tokens`、`top_p` 等 |
| `retry` | dict | 否 | — | `max_retries` 最大重试次数，`delay` 重试间隔（秒） |

```yaml
api:
  model: gpt-4o
  lang: zh
  concurrency: 10
  params:
    temperature: 0.95
    max_tokens: 2048
  retry:
    max_retries: 3
```

## 策略编排

`strategies` 是数组，支持多个策略组合，按 `weight` 比例分配总生成量（总和应为 **1.0**）。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `strategies[].type` | string | 是 | 策略类型 |
| `strategies[].weight` | float | 否 | 策略权重（默认 1.0） |

三种策略概览：

| 策略 | 思路 | 关键参数 |
|------|------|----------|
| `topic_driven` | 按主题/领域指定生成范围，内部随机题型和难度 | `topics` + `total_count` |
| `seed_driven` | 基于少量种子数据扩增，学习格式和风格 | `seed_file` + `example_num` + `target_count` |
| `self_instruct` | 模型自主构思指令，无需外部数据 | `target_count` |

各策略具体参数见下方。

### topic_driven

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `topic_driven` |
| `weight` | float | 否 | 策略权重 |
| `topics` | array | 是 | 主题列表 |
| `topics[].topic` | string | 是 | 主题名称 |
| `topics[].weight` | int | 否 | 该主题的相对占比（默认 1），各主题 weight 为相对值 |
| `topics[].knowledge` | string | 否 | 知识背景，引导模型生成准确内容 |
| `total_count` | int | 是 | 该策略总生成条数 |

内部通过模板随机化 **8 种题型**（概念解释、对比分析、应用题等）和 **3 级难度**（入门/进阶/高级）提升多样性。

```yaml
- type: topic_driven
  weight: 0.5
  topics:
    - topic: "Python 编程基础"
      weight: 3          # 占比 3/(3+2+1) ≈ 50%
      knowledge: "Python 是动态类型语言，支持面向对象、函数式编程。"
    - topic: "机器学习"
      weight: 2          # 占比 2/(3+2+1) ≈ 33%
    - topic: "数据库与 SQL"
      weight: 1          # 占比 1/(3+2+1) ≈ 17%
  total_count: 100
```

### seed_driven

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `seed_driven` |
| `weight` | float | 否 | 策略权重 |
| `seed_file` | string | 是 | 种子数据路径（JSONL），格式见 [seeds.jsonl](../seeds.jsonl) |
| `example_num` | int | 否 | 每批参考的样例数（默认 3） |
| `target_count` | int | 是 | 目标生成条数 |
| `field_map` | dict | 否 | 字段映射，如 `{instruction: question, output: response}` |

```yaml
- type: seed_driven
  weight: 0.3
  seed_file: ./seeds.jsonl
  example_num: 2
  target_count: 30
```

### self_instruct

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `self_instruct` |
| `weight` | float | 否 | 策略权重 |
| `target_count` | int | 是 | 目标生成条数 |

> self_instruct 固定串行执行，不受 `concurrency` 影响（每次生成依赖前一次的 seen_instructions 去重）。

```yaml
- type: self_instruct
  weight: 0.2
  target_count: 20
```

## 质量校验

生成阶段实时过滤低质量样本（Chain of Responsibility：长度 → 截断 → 去重）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `instruction_min_len` | int | 5 | 指令最小字符数 |
| `instruction_max_len` | int | 2000 | 指令最大字符数 |
| `output_min_len` | int | 30 | 输出最小字符数 |
| `output_max_len` | int | 6000 | 输出最大字符数 |
| `dedup` | bool | false | 是否实时去重 |
| `remove_truncated` | bool | false | 是否移除截断数据 |

```yaml
quality:
  instruction_min_len: 5
  instruction_max_len: 2000
  output_min_len: 30
  output_max_len: 6000
  dedup: true
  remove_truncated: true
```

## 数据清洗

生成后离线清洗，可独立通过 `clean` 命令调用。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `remove_html` | bool | true | 去除 HTML 标签 |
| `remove_urls` | bool | true | 去除 URL |
| `remove_emails` | bool | true | 去除邮箱 |
| `max_special_char_ratio` | float | 0.3 | 特殊字符最大占比 |
| `max_word_repetition_ratio` | float | 0.5 | 词汇重复最大占比 |
| `max_char_repetition_ratio` | float | 0.5 | 字符重复最大占比 |
| `dedup` | bool | true | 文本指纹去重（SHA256） |
| `instruction_min_len` | int | 5 | 清洗阶段指令最小长度 |
| `output_min_len` | int | 30 | 清洗阶段输出最小长度 |
| `field_map` | dict | — | 字段映射 |

```yaml
cleaner:
  remove_html: true
  remove_urls: true
  remove_emails: true
  max_special_char_ratio: 0.3
  max_word_repetition_ratio: 0.5
  max_char_repetition_ratio: 0.5
  dedup: true
```

> 清洗阶段与 quality 阶段的长度校验字段名相同但独立生效——quality 控制生成时过滤，cleaner 控制离线清洗时过滤。

### 语义去重

基于 embedding 向量的语义相似度去重，默认关闭。需额外的 embedding API 支持。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_dedup` | bool | false | 启用语义去重 |
| `embedding_model` | string | `text-embedding-3-small` | embedding 模型 |
| `embedding_api_key` | string | `$EMBEDDING_API_KEY` | 独立 embedding API 密钥（不填则回退到 `$API_KEY`） |
| `embedding_base_url` | string | `$EMBEDDING_BASE_URL` | 独立 embedding API 端点（不填则回退到 `$BASE_URL`） |
| `embedding_similarity_threshold` | float | 0.85 | 余弦相似度阈值，≥ 此值视为重复（建议 0.85~0.92） |
| `embedding_batch_size` | int | 20 | 批处理并发数 |

> **注意**：语义去重会额外消耗 embedding API 调用额度。chat 和 embedding 可使用不同 API 提供商（如 chat 用 DeepSeek，embedding 用阿里 DashScope）。

```yaml
cleaner:
  dedup: false
  embedding_dedup: true
  embedding_model: "text-embedding-3-small"
  embedding_similarity_threshold: 0.88
  embedding_batch_size: 20
```

## LLM 评分

LLM-as-Judge 多维度打分，通过 `score` 命令调用。维度完全可自定义。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `enabled` | bool | 否 | false | 生成后自动评分（pipeline 内） |
| `model` | string | 否 | 复用 `api.model` | 评分模型 |
| `api_key` | string | 否 | 复用 `api.api_key` | 评分 API 密钥 |
| `base_url` | string | 否 | 复用 `api.base_url` | 评分 API 端点 |
| `lang` | string | 否 | `en` | 评分提示语言 |
| `concurrency` | int | 否 | 3 | 并行评分线程数 |
| `dimensions` | array | 是 | `[]` | 评分维度 |
| `dimensions[].name` | string | 是 | — | 维度标识（输出 key） |
| `dimensions[].label` | string | 否 | — | 维度显示名 |
| `dimensions[].description` | string | 否 | — | 维度说明 |
| `dimensions[].max_score` | int | 否 | 10 | 分值范围 1~N |
| `params` | dict | 否 | — | LLM 调用参数 |
| `retry` | dict | 否 | — | 重试配置 |
| `min_total_score` | float | 否 | 0.0 | 最低总分阈值，低于此值被过滤出 `_scored_filtered.jsonl` |
| `output_path` | string | 否 | 自动拼接 `_scored.jsonl` | 评分结果输出路径 |
| `field_map` | dict | 否 | — | 字段映射 |

```yaml
scoring:
  enabled: true
  model: gpt-4o
  lang: zh
  concurrency: 3
  dimensions:
    - name: correctness
      label: "准确性"
      description: "答案是否准确无误，事实是否正确"
      max_score: 10
    - name: helpfulness
      label: "实用性"
      description: "答案是否对用户有实际帮助"
      max_score: 10
    - name: completeness
      label: "完整性"
      description: "答案是否完整全面，无遗漏"
      max_score: 10
    - name: clarity
      label: "清晰度"
      description: "表达是否清晰易懂，逻辑是否通顺"
      max_score: 10
  min_total_score: 20
```

输出每条数据附加 `scores` 和 `total_score`，可通过 `view` 命令查看评分分布。详见 [README.md](../README.md#数据查看)。

## 输出配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | — | 输出文件路径 |
| `format` | string | `alpaca` | 输出格式：`alpaca`（instruction/output）、`chatml`（messages）、`sharegpt`（conversations） |
| `multi_turn` | bool | false | 是否生成多轮对话（2-4 轮） |
| `checkpoint` | bool | false | 是否启用断点续传 |

### 输出流水线

```
生成阶段 → generated_sft.jsonl          # 原始生成数据
              ↓
清洗阶段 → generated_sft_cleaned.jsonl  # 去重 + 过滤后
              ↓
评分阶段 → generated_sft_scored.jsonl   # 附加 scores/total_score
              ↓
阈值过滤 → generated_sft_scored_filtered.jsonl  # 移除低于 min_total_score 的样本
```

> 每阶段保留中间文件，方便对比上下游数据差异。

```yaml
output:
  path: ./generated_sft.jsonl
  format: alpaca
  multi_turn: false
  checkpoint: false
```

## 全局配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dry_run` | bool | false | 预览模式，仅输出不调用 LLM |
| `count` | int | 100 | 预览模式下生成条数 |
| `random_seed` | int | null | 随机种子，固定后可复现结果 |

## 多样性提升

| 手段 | 配置 | 说明 |
|------|------|------|
| 提高 temperature | `api.params.temperature: 0.95` | 越大输出越随机（0.8~0.95） |
| 题型/难度随机化 | 模板内置 8 × 3 = 24 组合 | 每次生成随机选题型和难度 |
| knowledge 多样化 | `knowledge: "...请从不同角度提问"` | 引导模型变换提问角度 |
| 语义去重 | `cleaner.embedding_dedup: true` | 生成后清洗语义相似数据 |
| 多策略组合 | `strategies[]` 配置多种策略 | topic + seed + self_instruct 混合 |

## 失败重试

JSON 解析失败或 API 异常时自动重试。`_MAX_RETRIES = 4`（1 次初始 + 3 次重试），定义在 `alembic/strategies/base.py`。
