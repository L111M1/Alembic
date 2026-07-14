# 配置参考

Alembic 通过 YAML 文件控制生成、清洗、评分的全部行为。

## 完整示例

```yaml
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
      - topic: "机器学习"
        weight: 2
      - topic: "数据库与 SQL"
        weight: 1
    total_count: 100

  - type: evol_instruct
    weight: 0.3
    seed_file: ./seeds.jsonl
    max_rounds: 3
    depth_rate: 0.7
    branch_factor: 1

  - type: seed_driven
    weight: 0.2
    seed_file: ./seeds.jsonl
    target_count: 20

quality:
  instruction_min_len: 5
  instruction_max_len: 2000
  output_min_len: 30
  output_max_len: 6000
  dedup: true
  remove_truncated: true

cleaner:
  max_special_char_ratio: 0.3
  max_word_repetition_ratio: 0.5
  dedup: true
  embedding_dedup: false

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
| `lang` | string | 否 | `en` | 生成/提示语言（`zh`/`en`） |
| `concurrency` | int | 否 | 1 | 并行 API 调用数 |
| `params` | dict | 否 | — | LLM 参数：`temperature`（0.8~0.95）、`max_tokens` 等 |
| `retry` | dict | 否 | — | `max_retries`、`initial_delay`、`backoff_multiplier`、`max_delay` |

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

`strategies` 是数组，支持多个策略组合，按 `weight` 比例分配（总和应为 **1.0**）。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `strategies[].type` | string | 是 | 策略类型 |
| `strategies[].weight` | float | 否 | 策略权重（默认 1.0） |

四种策略概览：

| 策略 | 思路 | 关键参数 |
|------|------|----------|
| `topic_driven` | 按主题/领域指定生成范围，内部随机题型和难度 | `topics` + `total_count` |
| `seed_driven` | 基于少量种子数据扩增，学习格式和风格 | `seed_file` + `target_count` |
| `evol_instruct` | 迭代指令进化（Evol-Instruct），多轮深度/广度变异使指令逐步复杂 | `seed_file` + `max_rounds` |

### topic_driven

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `topic_driven` |
| `weight` | float | 否 | 策略权重 |
| `topics` | array | 是 | 主题列表 |
| `topics[].topic` | string | 是 | 主题名称 |
| `topics[].weight` | int | 否 | 该主题的相对占比（默认 1） |
| `topics[].knowledge` | string | 否 | 知识背景，引导模型生成准确内容 |
| `total_count` | int | 是 | 该策略总生成条数 |

### seed_driven

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | `seed_driven` |
| `weight` | float | 否 | 策略权重 |
| `seed_file` | string | 是 | 种子数据路径（JSONL） |
| `example_num` | int | 否 | 每批参考的样例数（默认 3） |
| `target_count` | int | 是 | 目标生成条数 |
| `field_map` | dict | 否 | 字段映射，如 `{instruction: question, output: response}` |
| `evolution` | dict | 否 | 进化配置（交叉/变异），见下方 |

#### evolution — 交叉与变异

借鉴遗传算法思想，在 few-shot 基础上增加两种算子。每次生成按轮盘赌选择模式：
`crossover_rate` 概率走交叉 → `mutate_rate` 走变异 → 其余走默认 few-shot。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `crossover_rate` | float | 0.0 | 交叉概率（0.0~1.0），与 `mutate_rate` 之和 >1 时自动归一化 |
| `mutate_rate` | float | 0.0 | 变异概率（0.0~1.0） |
| `crossover_mode` | string | `instruction_output` | 交叉模式：`instruction_output`（A 出指令 + B 出输出风格）或 `compose`（合并主题） |
| `mutation_types` | array | — | 变异类型定义列表，**必须显式配置** |

**`mutation_types[]` 条目格式：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 否 | 变异类型名（默认 `custom`） |
| `prompt` | string | 是 | 变异指令模板，支持 `{value}` 占位符 |
| `values` | array | 否 | 值池，每次随机选一个填入 `{value}` |
| `override_field` | string | 否 | 同步覆盖模板字段：`difficulty` 或 `question_type` |

```yaml
- type: seed_driven
  weight: 0.3
  seed_file: ./seeds.jsonl
  target_count: 100
  evolution:
    crossover_rate: 0.3
    mutate_rate: 0.3
    mutation_types:
      - name: difficulty
        values: [beginner, intermediate, advanced]
        prompt: "Change the difficulty to '{value}'"
      - name: tone
        values: [formal, casual]
        prompt: "Rewrite in a {value} tone"
```

### evol_instruct

迭代式指令进化（Evol-Instruct，基于 WizardLM），两阶段：多轮进化 → 回答生成。

**Phase 1 — 进化**：种子指令经过 N 轮深度/广度变异，逐轮筛选有效进化。

**Phase 2 — 回答**：每条进化后的指令独立调用 LLM 生成输出。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `type` | string | 是 | — | `evol_instruct` |
| `weight` | float | 否 | 1.0 | 策略权重 |
| `seed_file` | string | 是 | — | 种子数据路径（JSONL） |
| `field_map` | dict | 否 | — | 字段映射 |
| `max_rounds` | int | 否 | 3 | 最大进化轮数（≥1）。每轮深度/广度变异作用于上一轮产物，形成进化链 |
| `depth_rate` | float | 否 | 0.7 | 每轮每个指令走深度进化的概率（0.0~1.0）。剩余概率跳过深度进化 |
| `branch_factor` | int | 否 | 1 | 每轮每个指令的广度进化分支数。0 表示禁广度进化 |
| `depth_mutations` | array | 否 | 4 种默认算子 | 深度变异类型列表，每项 `{name, prompt}`。默认见下方 |
| `min_evolution_ratio` | float | 否 | 0.5 | 进化后与原指令的最小长度比。低于此值视为退化，丢弃 |
| `max_evolution_ratio` | float | 否 | 5.0 | 进化后与原指令的最大长度比。超过此值视为过度膨胀，丢弃 |
| `generate_output` | bool | 否 | true | 是否生成回答。设为 false 则只产出进化后的指令 |
| `require_reasoning` | bool | 否 | false | 回答时是否要求输出逐步推理链（在 JSON 中附加 `reasoning` 字段） |
| `include_seeds` | bool | 否 | false | 最终输出中是否包含原始种子（round=0，type=seed） |
| `evol_concurrency` | int | 否 | 1 | 进化阶段的并行数。与 `api.concurrency` 独立 |
| `evol_temperature` | float | 否 | 0.8 | 进化调用的 temperature |
| `evol_max_tokens` | int | 否 | 1024 | 进化调用的 max_tokens |
| `answer_temperature` | float | 否 | 0.6 | 回答生成的 temperature |
| `answer_max_tokens` | int | 否 | 2048 | 回答生成的 max_tokens |

**默认深度变异算子**（当 `depth_mutations` 未配置时自动使用）：

| name | 作用 |
|------|------|
| `add_constraint` | 添加具体约束或要求 |
| `deepen` | 增加查询的深度和广度 |
| `concretize` | 将一般概念替换为更具体的概念 |
| `increase_reasoning` | 要求显式的多步推理 |

**自定义深度变异**：

```yaml
- type: evol_instruct
  seed_file: ./seeds.jsonl
  max_rounds: 3
  depth_rate: 0.7
  branch_factor: 1
  depth_mutations:
    - name: add_constraint
      prompt: "Add one or more specific constraints or requirements"
    - name: domain_shift
      prompt: "Rewrite from the perspective of a specific industry (e.g., finance, healthcare)"
```

**元数据**：每条生成样本携带完整进化链信息：

```json
{
  "evolution_round": 3,
  "evolution_type": "depth",
  "mutation": "add_constraint",
  "evolution_chain": [
    "原始种子指令",
    "第一轮进化后的指令",
    "第二轮进化后的指令",
    "第三轮进化后的指令"
  ]
}
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

## 数据清洗

生成后离线清洗，可独立通过 `clean` 命令调用。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_special_char_ratio` | float | 0.3 | 特殊字符最大占比 |
| `max_word_repetition_ratio` | float | 0.5 | 词汇重复最大占比 |
| `max_char_repetition_ratio` | float | 0.5 | 字符重复最大占比 |
| `dedup` | bool | true | 文本指纹去重（MinHash） |
| `instruction_min_len` | int | 5 | 清洗阶段指令最小长度 |
| `output_min_len` | int | 30 | 清洗阶段输出最小长度 |
| `field_map` | dict | — | 字段映射 |

> 清洗阶段与 quality 阶段的长度校验字段名相同但独立生效——quality 控制生成时过滤，cleaner 控制离线清洗时过滤。

### 语义去重

基于 embedding 向量的语义相似度去重，默认关闭。需额外的 embedding API 支持。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_dedup` | bool | false | 启用语义去重 |
| `embedding_model` | string | `text-embedding-3-small` | embedding 模型 |
| `embedding_api_key` | string | `$EMBEDDING_API_KEY` | 独立 embedding API 密钥 |
| `embedding_base_url` | string | `$EMBEDDING_BASE_URL` | 独立 embedding API 端点 |
| `embedding_similarity_threshold` | float | 0.85 | 余弦相似度阈值，≥ 此值视为重复 |
| `embedding_batch_size` | int | 20 | 批处理并发数 |

## LLM 评分

LLM-as-Judge 多维度打分。维度完全可自定义。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `enabled` | bool | 否 | false | 生成后自动评分（pipeline 内） |
| `model` | string | 否 | 复用 `api.model` | 评分模型 |
| `api_key` | string | 否 | 复用 `api.api_key` | 评分 API 密钥 |
| `base_url` | string | 否 | 复用 `api.base_url` | 评分 API 端点 |
| `lang` | string | 否 | `en` | 评分提示语言 |
| `concurrency` | int | 否 | 3 | 并行评分线程数 |
| `dimensions` | array | 是 | — | 评分维度列表 |
| `dimensions[].name` | string | 是 | — | 维度标识（输出 key） |
| `dimensions[].max_score` | int | 否 | 10 | 分值范围 1~N |
| `params` | dict | 否 | — | LLM 调用参数 |
| `retry` | dict | 否 | — | 重试配置 |
| `min_total_score` | float | 否 | 0.0 | 最低总分阈值，低于此值被过滤 |
| `output_path` | string | 否 | 自动拼接 | 评分结果输出路径 |
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
      max_score: 10
    - name: helpfulness
      label: "实用性"
      max_score: 10
    - name: completeness
      label: "完整性"
      max_score: 10
    - name: clarity
      label: "清晰度"
      max_score: 10
  min_total_score: 20
```

## 输出配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | — | 输出文件路径 |
| `format` | string | `alpaca` | `alpaca`（instruction/output）、`chatml`（messages）、`sharegpt`（conversations） |
| `multi_turn` | bool | false | 是否生成多轮对话 |
| `checkpoint` | bool | false | 是否启用断点续传 |

**输出流水线**：

```
生成阶段 → generated_sft.jsonl
              ↓
清洗阶段 → generated_sft_cleaned.jsonl
              ↓
评分阶段 → generated_sft_scored.jsonl
              ↓
阈值过滤 → generated_sft_scored_filtered.jsonl
```

## 全局配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dry_run` | bool | false | 预览模式，仅输出不调用 LLM |
| `count` | int | 100 | 总生成条数 |
| `random_seed` | int | — | 随机种子，固定后可复现结果 |

## 多样性提升

| 手段 | 配置 | 说明 |
|------|------|------|
| 提高 temperature | `api.params.temperature: 0.95` | 越大输出越随机（0.8~0.95） |
| 题目/难度随机 | 模板内置组合 | 每次生成随机选题型和难度 |
| knowledge 多样化 | `knowledge: "...请从不同角度提问"` | 引导模型变换提问角度 |
| **迭代进化** | **`evol_instruct`** | **Evol-Instruct 逐轮变异产生渐进复杂度** |
| 种子交叉/变异 | `seed_driven.evolution` | 遗传算法式算子 |
| 语义去重 | `cleaner.embedding_dedup: true` | 生成后清洗语义相似数据 |
| 多策略组合 | `strategies[]` 配置多种策略 | topic + seed + evol 混合 |

## 失败重试

API 调用、JSON 解析失败时自动重试，指数退避。配置项：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_retries` | int | 3 | 最大重试次数 |
| `initial_delay` | float | 1.0 | 首次重试等待秒数 |
| `backoff_multiplier` | float | 2.0 | 退避倍数 |
| `max_delay` | float | 30.0 | 最大等待秒数 |

重试覆盖 API 层、策略生成层、规划层、进化回答层、评分层，统一使用 `retry_with_backoff()`（`alembic/api/base.py`）。
