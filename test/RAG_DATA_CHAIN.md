# 真实 RAG 评测数据链路准备

本阶段基线为 `bd0230f2f14b03fa016829cf895a4b009515e88c`。新增的是可执行导出工具和离线完整性校验，**没有运行真实后端检索或真实效果实验**，没有读取/配置凭据、调用付费模型、启动服务栈或使用 GPU。

## 当前材料状态

[40 题审阅册](review_materials/stage2/REVIEW.md)、[机器可读待审题目](review_materials/stage2/questions.jsonl)和[缺口清单](review_materials/stage2/readiness.json)均已生成。40 题全部 `pending_review`，人工审定 0 题，旧候选块共 200 个，仓库中可提供的对应原文为 0。

检查基线 Git 跟踪文件可见：`test/test.txt` 没有提交；`gold.txt`、生成脚本和待审清单只有问题、来源和候选 ID；`test/fixtures/rag_eval` 是合成文本。项目自己的 README/架构文档与这些掘金文章不是同一原文，不能拿来填充。来源 URL 只说明待补材料在哪里，不证明文章当前存在或内容与当年版本一致。本轮没有重新爬取网页。

生成器旧字段 `chunk_index` 表示抽样前的文章块序号；Go 摄入 `chunk_idx` 是实际写入 PG 时的序号。两者不能直接相等，也不能仅凭内容哈希猜 PGID。需要原始摄入记录或受控重建评测索引后的实际身份回执。本工具不负责重新摄入或重建索引。

## 1. 冻结原文与实际 PG 身份

准备两个真实 JSONL 文件，以下仅为字段示意，不能作为真实材料：

```json
{"chunk_id":"canonical-child-id","doc_hash":"actual-ingestion-doc-hash","chunk_idx":0,"content":"原始子块全文","source_url":"https://source.example/article","content_sha256":"optional-actual-content-hash"}
```

原文必须带 canonical `chunk_id`、实际摄入 `doc_hash/chunk_idx`、完整子块内容与 `source_url`。可保留文档版本、原始 URL、获取时间等额外来源字段。不得重新切分原文却沿用旧标签 ID。`content_sha256` 可省略，由冻结工具计算；若提供则必须匹配。

PG JSONL 每行是 `pg_id, doc_hash, chunk_idx, content`。导出命令从 `rag_chunks` 读取真正的 child `id/content`，采用只读、repeatable-read 事务，不读取父块、不导出向量。它不运行应用 bootstrap，不创建表、索引或 collection，也不调用模型：

```console
# 仅编译；从干净 Git checkout 构建，保留 Go VCS 元数据。
go build -p 1 -o /path/to/rag-export ./cmd/rag-export
# 仅在已有只读评测数据库可用并获准访问后运行，输出文件请使用新路径：
/path/to/rag-export --backend-config /private/backend.json --allow-backend-read --snapshot > pg-export.jsonl
```

`rag_backend_config.example.json` 是设置字段模板。连接信息来自操作者已有环境；密码/API key 只通过环境变量名引用，不填入 JSON、不复制聊天中的 key。本阶段没有设置这些变量。不要用应用启动命令代替导出命令，不要让评测指向仍在写入的生产索引。

```console
python test/rag_data.py freeze --corpus original-corpus.jsonl --pg-export pg-export.jsonl --index-snapshot FROZEN_INDEX_ID --output-dir frozen
python test/rag_data.py validate --bundle frozen
```

冻结输出 `corpus.jsonl`、`pg_snapshot.jsonl`、`mapping.jsonl`、`manifest.json`，已有输出目录会被拒绝。采用 `(doc_hash, chunk_idx)` 精确关联，再逐字符校验 child 内容；重复 PGID、重复摄入身份、缺行、多出的异语料行、空原文或内容漂移都会失败。同文不同来源不会按内容合并。全量 PG 快照必须与本次评测语料一一对应，因此需使用已经隔离的评测索引。

manifest 固定三个文件的 SHA256 与记录数。`index_snapshot` 是操作者给出的冻结 ES/Milvus 索引标识，**不是工具对远端索引不可变性的证明**。必须保留实际快照/版本记录、embedding 模型来源及索引设置；工具能检查命中 PGID 和实时子块内容，无法仅凭名称证明向量、倒排配置或未命中的数据从未变化。

## 2. 为待审题目导出三组原始检索

先让待审题目绑定冻结语料，原始清单保持不变：

```console
python test/rag_data.py review --questions test/rag_review_queue.jsonl --bundle frozen --output-dir review-with-evidence
```

有原文的候选标成 `available_unreviewed`，仍无相关性结论；缺失候选保留 `missing_source` 和空内容。人工可以改写过宽问题、补充候选，之后应重新冻结题目文件并保留变更记录。

按 `rag_backend_experiment.example.json` 填写实验配置：语料/映射/PG 快照哈希从 manifest 复制；`revision` 必须等于实际 exporter 的干净构建 Git SHA；记录可执行文件和 private backend JSON 的 SHA256、精确 embedding/rerank 模型名、候选数、RRF 常数和预览长度。重排 prompt 版本由构建 Git SHA 固定，温度固定为 0。没有任何凭据出现在实验配置或报告中。

**下列命令会连接已有后端并调用 embedding/rerank；本阶段未执行，运行前另行确认服务访问与模型预算。** 同时需要显式 `--allow-model-calls`，默认拒绝执行：

```console
python test/rag_backend_adapter.py --questions review-with-evidence/questions.jsonl --config experiment.json --bundle frozen --executable /path/to/rag-export --backend-config /private/backend.json --output-dir unscored-runs --allow-model-calls
```

Go 命令直接装配只读 PG/Milvus/ES 客户端和现有 `Engine.RetrieveForEvaluation`。仅请求 `vector`、`hybrid`、`hybrid_rerank`，固定 top-5、关闭改写/KG/答案生成；不初始化索引。执行前核验预期代码版本及模型配置，成功结果带实际配置、问题指纹、PGID、实际摄入 doc_hash/chunk_idx 和原始 child 内容 SHA256；取回身份时再次核对内容，变动即失败。

Python 适配器在创建子进程前检查冻结文件、可执行文件和设置哈希；通过 PGID 显式映射 canonical ID，并拒绝内容漂移、未知/重复 PGID、组别/问题/配置错配。子进程不用 shell，每次请求 45 秒上下文、60 秒进程超时。已有仓储部分接口没有 context，单个后端查询可能超过内部期限，外层进程超时是当前最后边界。

三组按题目轮换执行顺序，输出 `vector.json`、`hybrid.json`、`hybrid_rerank.json`，**不生成评分报告，不升级 label_status**。失败题保留空排名和错误类型，不保存可能含敏感内容的 SDK stderr。`latency_ms` 包含本地校验、子进程启动及连接开销；成功项的 `backend_latency_ms` 是连接建立后检索/embedding/重排耗时。成本与用量未采集，不能宣称为 0。重复进程启动不是生产在线性能基准。

## 3. 盲审与评分闭环

```console
python test/rag_review.py pool --questions review-with-evidence/questions.jsonl --corpus frozen/corpus.jsonl --runs unscored-runs/vector.json unscored-runs/hybrid.json unscored-runs/hybrid_rerank.json --output packet.json
python test/rag_review.py template --packet packet.json --reviewer reviewer-a --output review-a.jsonl
python test/rag_review.py template --packet packet.json --reviewer reviewer-b --output review-b.jsonl
# 两位实际审阅者完成证据判断后：
python test/rag_review.py reconcile --packet packet.json --reviews-a review-a.jsonl --reviews-b review-b.jsonl --output reviewed.jsonl
python test/rag_eval.py validate --dataset reviewed.jsonl --corpus frozen/corpus.jsonl
```

有分歧需人工裁决，保留原始提交；候选存在、两个标识不同或自动一致均不等于已认证真人。因标注改变了文件哈希，正式评分前须针对 `reviewed.jsonl` 重新运行导出，禁止改写旧导出哈希冒充新实验，然后离线调用 `rag_eval.py compare`。`rag_backend_adapter:retrieve` 也兼容原有 `collect` 接口，需显式已有环境路径 `KNOWFLOW_EVAL_BUNDLE/EXPORTER/BACKEND_CONFIG` 及 `KNOWFLOW_EVAL_ALLOW_MODEL_CALLS=1`；未设置时拒绝执行。

## 本轮验证边界

本轮测试只针对新增数据链路：Go 使用 fake retriever 和 SQL mock；Python 使用临时合成文件与模拟子进程。覆盖只读快照失败不返回半截、真实 PGID/child 哈希、配置与版本不符、冻结完整性、未知/重复 ID、超时、未授权不启动进程、待审导出到盲审池且拒绝评分等。它们验证协议和失败行为，不证明真实连接可用或检索效果。

本轮 8 项 Go 测试、13 项 Python 测试通过，命令与原始日志见 [数据链路验证记录](results/stage2_data_chain/validation.json)。复现仅需已有 Go 依赖缓存和 Python 标准库：

```console
go test -p 1 -count=1 -timeout 30s ./cmd/rag-export
python -m unittest discover -s test -p test_rag_data.py -v
```

没有重跑上一阶段九项历史故障对照。旧的 0/9 → 8/9 → 9/9 仍只是合成错误传播契约；本阶段没有新增 Recall/MRR 提升数字。尚缺：40 题对应原文及实际摄入身份、已冻结的评测后端、已有凭据环境、独立人工证据审阅，以及预算允许后的真实三组运行。
