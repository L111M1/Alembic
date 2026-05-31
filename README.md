# Alembic - SFT Data Generation & Cleaning Pipeline

基于 LLM API 的轻量级 SFT 训练数据生成和清洗管线。命名源自蒸馏器（alembic），寓意将原始数据蒸馏提纯。

## 安装

```bash
pip install openai click pyyaml jinja2
```

## 快速开始

### 1. 设置环境变量

```bash
export API_KEY=sk-xxx
export BASE_URL=https://your-endpoint/v1
```

### 2. 编辑配置文件

```yaml
# sft_gen_config.yaml
api:
  model: qwen-plus
  lang: en
  concurrency: 4               # 并行 API 调用数
  params:
    temperature: 0.8
    max_tokens: 2048
  retry:
    max_retries: 3

strategies:
  - type: topic_driven
    weight: 0.5
    topics:
      - topic: "Python programming"
        weight: 3
      - topic: "Machine learning"
        weight: 3
    total_count: 60

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

cleaner:
  remove_html: true
  remove_urls: true
  remove_emails: true
  instruction_min_len: 5
  output_min_len: 30
  dedup: true

output:
  path: ./generated_sft.jsonl
  format: alpaca
```

### 3. 运行

```bash
# 生成 + 清洗一键执行
python -m alembic.cli generate --config sft_gen_config.yaml

# 预览模式
python -m alembic.cli generate --config sft_gen_config.yaml --dry-run --count 5

# 独立清洗已有数据集
python -m alembic.cli clean ./raw_data.jsonl

# 查看模板
python -m alembic.cli list-templates
```

## 并发控制

设置 `concurrency` 控制并行 API 调用数：

```yaml
api:
  concurrency: 8
```

TopicDriven 和 SeedDriven 策略支持并行，SelfInstruct 固定串行（每次生成依赖前一次输出）。`concurrency` 同时受 API rate limit 约束，建议不超过 provider 的并发上限。

## 三种生成策略

| 策略 | 适用场景 | 关键参数 |
|---|---|---|
| `topic_driven` | 明确的领域覆盖需求 | `topics` + `total_count` |
| `seed_driven` | 有少量高质量种子数据 | `seed_file` + `example_num` + `target_count` |
| `self_instruct` | 多样性、自主探索 | `target_count` |

## 数据清洗

`cleaner` 模块独立清洗 JSONL 数据集：

```bash
python -m alembic.cli clean input.jsonl -o output.jsonl
```

清洗步骤：HTML 标签去除 → URL 去除 → 邮箱去除 → 长度过滤 → 特殊字符过滤 → 去重

## API 支持

所有 OpenAI 兼容接口。配置或环境变量设置：

```bash
export API_KEY=sk-xxx
export BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 项目结构

```
alembic/
├── api/                   # API 适配层
├── cleaner/               # 数据清洗
├── prompts/               # 提示词系统
├── strategies/            # 生成策略
├── quality/               # 质量校验
├── writers/               # 数据输出
├── core/                  # 管线编排
├── config.py              # 配置解析
└── cli.py                 # CLI 入口
```

## License
