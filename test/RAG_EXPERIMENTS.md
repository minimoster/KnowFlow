# 离线 RAG 实验归档与配对复盘

`rag_experiment.py` 把**已经存在的三组原始检索导出**整理成可复核的实验目录。它调用现有 `rag_eval.compare` 评分，不引入新的检索实现、模型调用或评测框架。

本次增量提供归档、对照和内部一致性校验；没有运行真实模型、连接用户数据库，也没有得到新的 RAG 效果成绩。历史 SciFact 实验只有在对应数据、配置和原始导出被提供后才能另行复核。既有 40 道待审题仍沿用原来的审阅门禁，不能因为新增了报告功能而升级为已审数据。

## 1. 直接运行离线演示

在仓库根目录运行，Python 仅使用标准库：

```bash
python test/rag_experiment.py pack \
  --dataset test/fixtures/rag_eval/labels.jsonl \
  --corpus test/fixtures/rag_eval/corpus.jsonl \
  --runs test/fixtures/rag_eval/vector.json \
         test/fixtures/rag_eval/hybrid.json \
         test/fixtures/rag_eval/hybrid_rerank.json \
  --baseline vector \
  --fixture \
  --output-dir .data/rag-experiments/demo-001
```

输出显著标记 `OFFLINE_FIXTURE_NOT_REAL_RETRIEVAL`。示例三组本来就使用相同合成排序，只验证计算、归档和命令行为，不能用于简历中的效果提升数字。

**一条校验入口：**

```bash
python test/rag_experiment.py verify --bundle .data/rag-experiments/demo-001
```

成功返回 `integrity: consistent`，表示文件与清单相符，并且从原始导出重算的评分、配对结果和 Markdown 与保存内容一致。验证过程只读归档，临时重算文件由 Python 自动清理。

脚本使用两个阶段：先复制输入到临时目录并完成验证，再独占创建最终输出目录。输入文件不被改写。**输出必须是新目录，默认拒绝覆盖**；需要重新运行时换一个目录名。如果文件系统写入途中失败，未完成目录不会通过 `verify`，应保留或移走该目录后使用新名称重试。

## 2. 真实导出如何进入归档

先按 [RAG 数据链路准备](RAG_DATA_CHAIN.md) 固定语料、索引身份和 PGID 映射，按 [RAG 评测说明](RAG_EVALUATION.md) 完成证据标签检查并取得三组导出。归档工具不执行 `collect`、不加载适配器、不发起任何付费调用。

已有原始导出后：

```bash
python test/rag_experiment.py pack \
  --dataset PATH_TO_REVIEWED_LABELS.jsonl \
  --corpus PATH_TO_FROZEN_CORPUS.jsonl \
  --runs PATH_TO_VECTOR.json PATH_TO_HYBRID.json PATH_TO_HYBRID_RERANK.json \
  --baseline vector \
  --output-dir .data/rag-experiments/experiment-001
```

真实归档不加 `--fixture`，要求：

- 三组 `kind=retrieval`，题目 `label_status=reviewed`，并保留既有审阅人、时间、原文证据及哈希检查。
- 原始导出的 `config.revision` 是实际受测版本的完整 40 位小写 commit SHA；模型、索引和重排标识不能保留 `REPLACE_...` 或 `TODO` 占位符。这里检查格式，不声称远程认证该 commit 或模型确实运行过。
- 同一次归档只包含 `development` 或 `test` 一种切分。开发集用于选择参数；测试集独立冻结并分别导出，不混合汇总。
- 三组的题目、语料和共有配置相同；只允许原评测协议规定的模式/重排差异。失败题必须保留且导出空排名与错误信息。
- `cost_usd` 可为非负有限数字，未采集则为 `null` 或缺省。不能填零来表示未知。

外部公开基准若使用其他标签协议，应先明确映射关系并完善对应适配，再进入现有评分器；此工具不会把模型自动生成的标签改成“人工审定”。

## 3. 输出内容与可追溯信息

| 文件 | 用途 |
| --- | --- |
| `labels.jsonl`、`corpus.jsonl` | 输入字节的原样副本，保留证据和切分 |
| `vector.json`、`hybrid.json`、`hybrid_rerank.json` | 三组原始逐题导出，保留配置、排名、失败、时延和已采集用量 |
| `report.json` | 由现有 `rag_eval.compare` 生成的逐题评分与汇总 |
| `comparison.json` | 相对所选基线的逐题指标差值、错误转变、结果分类和成本覆盖 |
| `REPORT.md` | 中文可读报告：三组结果、配对差值、待复盘题目与适用边界 |
| `manifest.json` | 文件 SHA-256 与字节数、两个评分脚本的哈希、受测版本、基线、切分和结果类别 |

`verify` 不只检查文件哈希，还会重新执行既有评分器并重建报告。即使单独修改某个报告并更新其清单哈希，只要它与原始导出不一致，也会被拒绝。脚本版本发生变化时校验会明确拒绝，应使用清单对应的脚本版本复核，或从原始输入生成一个新目录，不修改旧归档。

这些机制检查**归档内部一致性**。清单没有外部签名，不是来源认证；同步更改输入和清单的人仍可构造一套自洽文件。它不证明模型真实运行、审阅者为真人、远端 ES/Milvus 索引未变化，或实验能够代表线上效果。

## 4. 如何切换对照基线

`--baseline` 支持 `vector`、`hybrid`、`hybrid_rerank`，默认 `vector`。例如关注“混合检索之后，重排是否值得增加”时，使用同一批三组导出，另建目录：

```bash
python test/rag_experiment.py pack \
  --dataset test/fixtures/rag_eval/labels.jsonl \
  --corpus test/fixtures/rag_eval/corpus.jsonl \
  --runs test/fixtures/rag_eval/vector.json \
         test/fixtures/rag_eval/hybrid.json \
         test/fixtures/rag_eval/hybrid_rerank.json \
  --baseline hybrid \
  --fixture \
  --output-dir .data/rag-experiments/demo-hybrid-baseline-001
```

切换基线只改变比较方向，不删除任何组或题目，不触发重跑，也不允许跨模型、跨语料或跨配置的不受控混合。

## 5. 如何读指标与失败复盘

- **Recall@5** 是每题已标相关证据中被 Top 5 找到的比例，再对所有题取平均。它不等于“至少命中一条”的 Hit@5。
- **MRR@5** 只看前五名中首个相关证据的位置；第五名之外不贡献分数。它不是全排名 MRR。
- 报告中的 `ΔRecall 百分点` 是候选减基线的比例差乘 100；`ΔMRR` 是原尺度差值；延迟差单位是毫秒。
- 逐题“改善/退步”要求 Recall 与 MRR 同向或一项不变；一升一降归为“权衡”，两项相同归为“不变”。
- `execution_transition` 单独记录首次错误恢复、引入新错误、两组均失败或均成功。错误恢复后仍然空检索时，零分不变，但错误恢复事件不会丢失。
- 结果分类是执行错误、空检索、未命中已标证据、部分命中、全部命中。它们帮助选择复盘题目，不能自动证明问题在 embedding、索引、改写或重排的哪一层。
- 所有失败题以零分计入原有分母，不能删掉它们提高指标。未标注候选仍按原协议视为不相关，应在解释结果时说明标注范围。
- 只有全部题目采集了成本，`total_usd` 才有数值。部分已知时给出覆盖题数和 `known_subtotal_usd`，总成本保持未知；原始 token 用量仍留在导出中，不推算未提供的 token。
- 耗时沿用导出器记录的 embedding/检索/重排调用范围，不包含入库、最终答案生成或完整 Agent 工作流。P95 使用原评分器的 nearest-rank 定义；少量样本的 P95 和一次对照都不能证明稳定线上性能。

## 6. 回归与回退

```bash
python -m unittest discover -s test -p 'test_rag*.py' -v
```

回归针对已有风险：待审标签误评分、fixture 混入真实成绩、缺题或配置漂移、原始导出与报告不一致、成本缺失误记零、错误恢复被零分掩盖、指标一升一降被误标为全面提升，以及覆盖已有目录。

本功能仅新增离线脚本和文档，不改变后端路由、检索策略、数据库结构、模型配置或服务启动行为。停用时继续使用原来的 `rag_eval.py compare` 即可。要回到升级前代码，可在没有未保存修改的独立工作区检出已保留的 `backup/pre-agent-upgrade-20261026` 分支；原 main 不需要被覆盖。已经生成的归档目录与代码版本独立保留，只有在明确不需要时才自行删除。
