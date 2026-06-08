# SFT 数据生成管线配置

```yaml
# 完整示例：sft_gen_config.yaml
api:
  model: deepseek-v4-flash
  api_key: ""
  base_url: ""
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

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | LLM 模型名称 |
| `api_key` | string | 否 | API 密钥（建议通过环境变量 `API_KEY` 设置） |
| `base_url` | string | 否 | API 端点（建议通过环境变量 `BASE_URL` 设置） |
| `lang` | string | 否 | 生成语言（zh/en） |
| `concurrency` | int | 否 | 并行 API 调用数 |
| `params` | dict | 否 | 传给 LLM 的参数字典，支持任意 OpenAI 兼容参数<br>`temperature`（建议 0.8~0.95 提高多样性）, `max_tokens`, `top_p` 等 |
| `retry` | dict | 否 | 重试配置<br>`max_retries`: 最大重试次数<br>`delay`: 重试间隔（秒） |

```yaml
# API 配置示例
api:
  model: gpt-4o
  api_key: ""              # 或通过环境变量 API_KEY 设置
  base_url: ""             # 或通过环境变量 BASE_URL 设置
  lang: zh
  concurrency: 10
  params:
    temperature: 0.95
    max_tokens: 2048
  retry:
    max_retries: 3
```

## 策略编排

`strategies` 是一个数组，支持同时配置**多个生成策略**，按 `weight` 比例分配总生成量。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `strategies[].type` | string | 是 | 策略类型：`topic_driven` / `seed_driven` / `self_instruct` |
| `strategies[].weight` | float | 否 | 策略权重，所有策略 weight 总和应为 **1.0** |

## 生成策略

### 1. topic_driven（主题驱动）

按指定**多个领域/主题**和知识范围生成数据，可灵活配比不同领域的生成量。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `topic_driven` |
| `weight` | float | 否 | 策略权重（默认 1.0），多个策略总和应为 1 |
| `topics` | array | 是 | **主题列表，支持多个领域** |
| `topics[].topic` | string | 是 | 领域/主题名称 |
| `topics[].weight` | int | 否 | 该领域在总生成中的**相对占比**（值越大生成越多，默认 1） |
| `topics[].knowledge` | string | 否 | 该领域的知识背景，指导模型生成高质量内容 |
| `total_count` | int | 是 | 该策略总生成条数 |

> 多个领域的 weight 为**相对值**，不要求总和为特定值。例如 `Python:3, ML:2, DB:1` 表示按 3:2:1 比例分配生成量。

```yaml
# topic_driven 多领域配比示例
- type: topic_driven
  weight: 0.5
  topics:
    - topic: "Python 编程基础"
      weight: 3          # 占比 3/(3+2+1) ≈ 50%
      knowledge: "Python 是动态类型语言，支持面向对象、函数式编程。"
    - topic: "机器学习"
      weight: 2          # 占比 2/(3+2+1) ≈ 33%
      knowledge: "三大范式：监督学习、无监督学习、强化学习。"
    - topic: "数据库与 SQL"
      weight: 1          # 占比 1/(3+2+1) ≈ 17%
      knowledge: "关系型：MySQL、PostgreSQL，ACID 事务。"
  total_count: 100
```

### 2. seed_driven（种子驱动）

基于已有种子数据扩增，权重占比建议 30%。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `seed_driven` |
| `weight` | float | 否 | 策略权重 |
| `seed_file` | string | 是 | 种子数据文件路径（JSONL） |
| `example_num` | int | 否 | 每批参考的样例数 |
| `target_count` | int | 是 | 目标生成条数 |

```yaml
# seed_driven 示例
- type: seed_driven
  weight: 0.3
  seed_file: ./seeds.jsonl
  example_num: 2
  target_count: 30
```

### 3. self_instruct（自我指令）

模型自主生成指令-回答对，无需外部数据，权重占比建议 20%。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `self_instruct` |
| `weight` | float | 否 | 策略权重 |
| `target_count` | int | 是 | 目标生成条数 |

```yaml
# self_instruct 示例
- type: self_instruct
  weight: 0.2
  target_count: 20
```

## 质量校验

| 字段 | 类型 | 说明 |
|------|------|------|
| `instruction_min_len` | int | 指令最小长度 |
| `instruction_max_len` | int | 指令最大长度 |
| `output_min_len` | int | 输出最小长度 |
| `output_max_len` | int | 输出最大长度 |
| `dedup` | bool | 是否去重 |
| `remove_truncated` | bool | 是否移除截断数据 |

```yaml
# 质量校验示例
quality:
  instruction_min_len: 5
  instruction_max_len: 2000
  output_min_len: 30
  output_max_len: 6000
  dedup: true
  remove_truncated: true
```

## 去重机制

去重分为两个阶段，分别配置在 `quality` 和 `cleaner` 中。

### 1. 实时去重（quality 阶段）

生成过程中实时校验，`quality.dedup: true` 时启用。

```
instruction + output → SHA256 → seen 集合 → 重复则丢弃
```

### 2. 离线去重（cleaner 阶段）

清洗阶段执行，支持两种模式：

**文本指纹去重**（`cleaner.dedup: true`，默认启用）

```
instruction + output → strip().lower() → SHA256 → seen 集合 → 重复则丢弃
```

**语义去重**（`cleaner.embedding_dedup: true`，默认关闭）

1. 将 instruction + output 拼接后调用 embedding 模型转向量
2. L2 归一化后计算余弦相似度（cosine similarity）：`sim = A·B / (|A|×|B|)`
3. 相似度 ≥ `embedding_similarity_threshold` 则视为重复，保留第一条
4. embedding 调用支持并发（ThreadPoolExecutor），批量加速

> embedding 可与 chat 使用不同的 API 服务（如 chat 用 DeepSeek，embedding 用阿里 DashScope），通过 `embedding_api_key` / `embedding_base_url` 单独配置。

```yaml
# 语义去重配置示例（独立 embedding API）
cleaner:
  dedup: false                          # 关闭文本指纹去重
  embedding_dedup: true                 # 启用语义去重
  embedding_model: "text-embedding-3-small"
  embedding_api_key: "sk-xxx"           # 独立 embedding API 密钥（可选）
  embedding_base_url: "https://api.openai.com/v1"  # 独立 embedding 端点（可选）
  embedding_similarity_threshold: 0.88  # ≥0.88 视为重复
  embedding_batch_size: 20              # 每批并发数
```

> 注意：语义去重会额外消耗 embedding API 的调用额度。

## 数据清洗

| 字段 | 类型 | 说明 |
|------|------|------|
| `remove_html` | bool | 去除 HTML 标签 |
| `remove_urls` | bool | 去除 URL |
| `remove_emails` | bool | 去除邮箱 |
| `max_special_char_ratio` | float | 特殊字符最大比例 |
| `max_word_repetition_ratio` | float | 词汇重复最大比例 |
| `max_char_repetition_ratio` | float | 字符重复最大比例 |
| `embedding_dedup` | bool | 基于 embedding 的语义去重 |
| `embedding_model` | string | embedding 模型名称 |
| `embedding_api_key` | string | 独立 embedding API 密钥（不填则复用 `api.api_key`） |
| `embedding_base_url` | string | 独立 embedding API 端点（不填则复用 `api.base_url`） |
| `embedding_similarity_threshold` | float | 语义去重阈值（默认 0.85，建议 0.85~0.92） |
| `embedding_batch_size` | int | embedding 批处理大小 |

```yaml
# 数据清洗示例
cleaner:
  remove_html: true
  remove_urls: true
  remove_emails: true
  max_special_char_ratio: 0.3
  max_word_repetition_ratio: 0.5
  max_char_repetition_ratio: 0.5
  dedup: true
  embedding_dedup: false
  embedding_model: "text-embedding-3-small"
  embedding_similarity_threshold: 0.85
  embedding_batch_size: 20
```

## LLM 评分

用于 `score` 命令，调用 LLM 对生成数据进行多维度打分。所有维度通过 YAML 配置文件定义。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `enabled` | bool | 否 | `false` | 生成后是否自动评分 |
| `model` | string | 否 | `gpt-4o` | 评分模型 |
| `api_key` | string | 否 | — | 评分 API 密钥（不填则复用 `api.api_key`） |
| `base_url` | string | 否 | — | 评分 API 端点（不填则复用 `api.base_url`） |
| `lang` | string | 否 | `en` | 评分提示语言（zh/en） |
| `concurrency` | int | 否 | `3` | 并行评分数 |
| `dimensions` | array | 是 | `[]` | 评分维度列表，每项含 `name`、`label`、`description`、`max_score` |
| `params` | dict | 否 | `{}` | LLM 调用参数 |
| `retry` | dict | 否 | `{}` | 重试配置 |
| `min_total_score` | float | 否 | `0.0` | 最低总分阈值，低于此值的样本被过滤 |
| `output_path` | string | 否 | — | 评分结果输出路径 |
| `field_map` | dict | 否 | — | 字段映射 |

```yaml
# LLM 评分配置示例
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
  params:
    temperature: 0.3
    max_tokens: 1024
  retry:
    max_retries: 3
  min_total_score: 0.0
```

## 输出配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `path` | string | 输出文件路径 |
| `format` | string | 输出格式（alpaca/sharegpt） |
| `multi_turn` | bool | 是否生成多轮对话（2-4轮） |
| `checkpoint` | bool | 是否启用断点续传 |

```yaml
# 输出配置示例
output:
  path: ./generated_sft.jsonl
  format: alpaca
  multi_turn: false
  checkpoint: false
```

### 输出流水

管线分两步执行，输出两个文件：

```
生成阶段 → generated_sft.jsonl          # 原始生成数据（含低质量、重复项）
                ↓
清洗阶段 → generated_sft_cleaned.jsonl  # 清洗后数据（去重、过滤）
```

1. **生成** → 写入 `output.path`（如 `generated_sft.jsonl`）
2. **清洗** → 读取生成文件，清洗后写入 `{path}_cleaned.jsonl`（如 `generated_sft_cleaned.jsonl`）

> 清洗阶段会保留原始生成文件，方便对比清洗前后的数据差异。

## 全局配置

顶层字段，控制管线行为。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dry_run` | bool | 否 | `false` | 预览模式，仅输出生成的指令不调用 LLM |
| `count` | int | 否 | `100` | 预览模式下生成条数 |
| `random_seed` | int | 否 | `null` | 随机种子，固定后可复现结果 |

## 提升多样性

| 手段 | 配置 | 说明 |
|------|------|------|
| 提高 temperature | `api.params.temperature: 0.95` | 越大输出越随机，建议 0.8~0.95 |
| knowledge 加多样性指令 | `knowledge: "...请从不同角度提问"` | 引导模型每次问不同方向 |
| 语义去重 | `cleaner.embedding_dedup: true` | 生成后清洗相似数据 |

## 失败重试

JSON 解析失败或 API 异常时自动重试（最多 3 次），避免因偶发错误丢失数据。配置在 `alembic/strategies/base.py` 的 `_MAX_RETRIES = 4`（1 次初始 + 3 次重试）。
