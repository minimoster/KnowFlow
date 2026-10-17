# KnowFlow 第一阶段：错误真实性与 RAG 评测基础

基线 main：`9afe774372f2068eb7e57ecdf1c491965eace2e3`。本阶段修改代码并运行离线测试，未调用真实模型、使用 GPU 或启动数据库/服务栈。

## 现在可以验证什么

- LLM 同步、快模型、流式调用统一返回 `(string, error)`。HTTP 429/5xx、超时、取消、无效/空响应都返回错误，真实模式不回退 mock。流式读取错误、缺失 `[DONE]` 会失败；已推送的 token 属于未完成内容，不发起第二次同步请求。
- `llm.mode` 默认 `real`；只有显式设为 `mock` 才允许演示回复。缺少凭据会报错。测试通过结构体 `LLMConfig{LLMMode: "mock"}` 显式开启。没有配置任何新凭据。
- RAG 生成、改写、重排和查询 embedding 接收调用方 context。子 Agent 和 prompt skill 返回真实错误。失败请求返回 `Response.error` / SSE `error`，不写回答案记忆；失败的上游子 Agent 会阻止后续报告保存；生成失败不会标记任务完成或物化交付文件。
- 可选的记忆提取/分类、规划和文件类型判断仍保留原有规则兜底或跳过行为；这些路径不会制造 mock 模型答案。
- 原有持久化 Repo 查询接口没有 context 参数，因此数据库查询/索引调用本身的即时中断仍受后端现有超时约束。本阶段不重构存储接口。查询结束会检查取消，随后不会继续 LLM 生成。

## 40 题证据审阅流程

`rag_review_queue.jsonl` 是从旧 50 题中取前 40 题建立的**开发集待审阅清单**。所有条目的 `label_status=pending_review`，`evidence=[]`，候选相关性为 `null`。没有声称已人工审定或已阅读不存在的原文。

当前仓库没有提交 `test/test.txt`；旧 `gold.txt` 的五块来自均匀抽样，相关性等级来自位置。现已明确标为 `legacy_unreviewed`，不能直接作为评测 gold。余下 10 题也没有经过审定，不能据此声称已有可信最终测试集。

1. 提供旧数据对应的冻结原文块 JSONL（每条至少有 `chunk_id`、`content`），保留 URL、文档版本等来源字段。不要在 MMI 目录内收集、修改或生成数据。离线评测不要求重新爬取网页。
2. 在独立目录重新生成清单，填入候选原文和内容哈希：

   ```console
   python test/rag_eval.py prepare --legacy test/gold.txt --corpus /path/to/corpus.jsonl --count 40 --output review.jsonl
   ```

3. 审阅者逐题阅读证据；将过宽的文章概述问题改成可由具体段落核验的问题。删除不支持答案的候选，补充其他文档中的有效证据。候选来源不是相关性判断。建议覆盖直接事实、多证据整合、近义表达、术语/代码精确匹配、易混淆文档。
4. 每个实际判定加入 `evidence`，写出 `chunk_id`、`content_sha256`、`start`、`end`、`quote`、`relevance`、`rationale`。偏移以 Python Unicode 字符计，`end` 为开区间，必须满足 `content[start:end] == quote`。等级：0 不支持；1 部分支持；2 直接支持；3 必需证据。本轮 Recall 将所有 >0 标签视作相关。
5. 审阅者实际完成判定后，填写真实 `reviewed_by`、ISO 日期 `reviewed_at`，并改成 `label_status=reviewed`；记录冻结语料文件的 SHA256。候选本身不会升级状态，不应把 Codex 生成的候选视为人工确认；下述双人工作流仅在完整提交一致时生成 reviewed，不能验证提交者是否真人。
6. 在开发集上先合并三组检索的候选池，审阅遗漏的有效证据，再冻结标注。未标注的命中在计算中按不相关处理，标注不完整会低估召回，应在报告中说明。正式测试集需另行冻结，避免反复调参污染。

   ```console
   python test/rag_eval.py validate --dataset review.jsonl --corpus /path/to/corpus.jsonl
   ```

语料哈希变化、证据 ID 缺失、引文/偏移不一致、未审阅或没有正例都会拒绝评分。本阶段只评估有答案题的检索；资料不足、冲突与拒答测试留在后续。

## 三组检索接口与重复实验

Go 的 `Engine.RetrieveForEvaluation(ctx, question, mode, candidateK)` 复用现有语义检索和 RRF 实现，仅做只读检索，不调用生成答案、不初始化索引或启动服务。支持 `vector`、`hybrid`、`hybrid_rerank`。固定关闭改写、KG，三组共用相同候选池参数，最终输出 Top 5。混合检索沿用生产逻辑，每路召回 `2*candidateK`，融合后保留 `candidateK`。

这个入口在真实后端/embedding 出错时返回错误，禁止静默降级成其他实验组；重排解析错误也不会被算成成功。结果保留数据库 child `PGID`。数据导入时需要保存 **PGID → corpus chunk_id** 映射及其哈希，禁止用父块展示内容或不唯一的内容哈希猜测 ID。

Python 支持两种入口：

- `compare`：读取已经导出的三个 JSON 文件，纯离线计算。
- `collect`：显式传入 `--adapter module:function` 才会执行检索。适配函数签名 `retrieve(question, mode, config) -> dict`；返回 `ranked_chunk_ids`（canonical child ID，最多 5 项）、可选 `error`、`usage`、`cost_usd`。它应调用上述 Go 接口或同等严格的检索导出入口，不得静默降级。适配代码、实际服务连接和预算准备好后再运行；本次未连接这些服务。

配置见 `rag_eval_config.example.json`。必须记录测试提交、精确模型/提示词版本、索引快照与 ID 映射哈希、TopK、候选池和 RRF 常数。文件中不放凭据。`collect` 逐题轮换三组执行顺序、测量耗时，失败题保留并计为零召回；异常只保存类型，防止 SDK 错误文字带出敏感信息。

```console
python test/rag_eval.py compare --dataset review.jsonl --corpus corpus.jsonl --runs vector.json hybrid.json hybrid_rerank.json --output report.json
# 仅在服务、适配器与付费预算已批准后：
python test/rag_eval.py collect --dataset review.jsonl --corpus corpus.jsonl --adapter my_adapter:retrieve --config experiment.json --output-dir results
```

导出文件格式见 `fixtures/rag_eval/vector.json`（其中 `kind=fixture`，不可当真实结果）。真实导出应使用 `kind=retrieval` 并匹配语料和题目文件哈希。三组的共有配置必须完全一致；只有 mode/reranker 变化。所有题都必须出现，不能删掉失败题或重复命中 ID。

报告包括每题 Top 5、命中的证据、Recall@5、MRR@5、错误、延迟和可选用量/成本，以及宏平均 Recall@5、宏平均 MRR@5、平均耗时和 nearest-rank p95。这里 **MRR@5 是截断指标**，不会误称全排名 MRR。计时包含 adapter 内的 embedding/检索/重排，不含导入索引和最终答案生成；未测成本为 null，不伪记 0。真实效果、成本和答案质量目前均未测量。

## 离线复现与五个故障案例

```console
python -m unittest discover -s test -p 'test_rag*.py' -v
python test/rag_eval.py fixture --output-dir test/fixtures/rag_eval
go test -p 1 -count=1 -timeout 90s ./internal/infrastructure/llm ./internal/application/chat ./internal/application/skill ./internal/domain/rag
```

受限 Windows 环境应把 `TEMP`/`TMP` 指向任务目录内的 `.tmp`，设置 `GOMAXPROCS=2`、`GOTOOLCHAIN=local`、`GOPROXY=off`，避免额外下载和过度并行。故障测试通过内存 HTTP transport 与仓储 fake 注入响应，不监听端口、不连接真实应用。

| 故障案例 | 改前风险 | 现在的离线验证 |
|---|---|---|
| HTTP 429/503（包括带 success-shaped body） | 普通调用回退 mock；stream 可能二次请求 | `TestHTTPFailureNeverFallsBackToMock`：空答案、真实错误、只发一次请求 |
| HTTP client 超时 | 返回模拟答案 | `TestTimeoutAndInFlightCancellation`：同步/stream 都返回 deadline error |
| 请求处理中取消 | 返回中断字符串却被当作成功；RAG 丢 context | 同上及 `TestRAGCallbacksReceiveTaskContextAndPropagateFailure`：三个回调收到任务身份与取消 |
| 流只返回半截便结束 | 忽略 scanner/EOF，可能把半截当完整答案 | `TestTruncatedStreamReturnsErrorWithoutReplay`：明确错误，不重放；已输出 token 保留为未完成 |
| writer 模型失败 | 子 Agent `(string,nil)`，后续可能保存伪报告 | `TestFailedWriterDoesNotRunDependentDocumentAgent`：writer failed，保存计数为 0；生成错误向响应传播 |

fixture 三组结果刻意相同，只验证指标计算与文件协议；`report.json` 显著标记 `OFFLINE_FIXTURE_NOT_REAL_RETRIEVAL`，任何数值都不能写成真实优化成绩。

后续数据链路已增加只读后端导出命令、冻结语料/ID 映射门禁和 Python 适配器，见 [RAG 数据链路准备](RAG_DATA_CHAIN.md)。尚待实际材料：原文语料及摄入身份、40 题证据人工审阅、已有冻结后端接入及经预算确认的三组运行。引用溯源、拒答和大架构改造不在本阶段。

## 双人证据审阅与失败复盘

`rag_review.py pool` 合并三组命中和已有候选，并隐藏组名、排名、分数与旧等级；候选按固定哈希顺序展示。首次建池需要由检索导出器提供针对**待审问题文件**的三组原始结果及其哈希，不需要先评分；`rag_eval.py collect` 要求已审标签，因此只用于审阅完成后的实验，不能拿它给未审题评分。新增 `rag_backend_adapter.py` 提供待审题目导出入口，详见数据链路说明；真实后端未在本轮连接。

```console
python test/rag_review.py pool --questions review.jsonl --corpus corpus.jsonl --runs vector.json hybrid.json hybrid_rerank.json --output packet.json
python test/rag_review.py template --packet packet.json --reviewer reviewer-a --output review-a.jsonl
python test/rag_review.py template --packet packet.json --reviewer reviewer-b --output review-b.jsonl
# 两位审阅者独立阅读、填写全部候选的等级/理由、正例精确引文及真实时间，完成后再执行：
python test/rag_review.py reconcile --packet packet.json --reviews-a review-a.jsonl --reviews-b review-b.jsonl --output reviewed.jsonl
python test/rag_eval.py validate --dataset reviewed.jsonl --corpus corpus.jsonl
python test/rag_review.py failures --report report.json --dataset reviewed.jsonl --corpus corpus.jsonl --output failures.jsonl
```

空白模板的 complete=false、等级和时间为空。重复审阅者标识、漏判、伪造引文拒绝合并；分歧或无正例产生 needs_adjudication，不能评分。人工裁决须保留两份原始提交及裁决理由，禁止按得分挑标签。合成输入即使意见一致也仍是 fixture。工具仅检查提交格式、一致性和证据定位，不认证真人身份。失败导出保留原始配置、召回内容和缺失证据，分析/行动字段留空待复盘。标注文件变化后必须重新导出匹配新哈希的结果，禁止把旧结果哈希改写成新实验。

## 已运行的九项合成故障契约对照

原始 Go JSONL 日志与汇总见 [stage1_contract/report.json](results/stage1_contract/report.json)。同一组内存 transport 注入的故障包括 429、503、超时、处理中取消、空响应、坏 JSON、截断流、响应返回时取消、缺少配置。

| 固定版本 | 契约通过数 |
|---|---:|
| `9afe774372f2068eb7e57ecdf1c491965eace2e3` | 0/9 |
| `c3e62adde05d5413771b8b16dcf87e47ef6ccdfa` | 8/9 |
| `bb0cdc16f8c99dc72b3257d666204ed8c6adafb5` | 9/9 |

契约要求失败/取消返回空最终答案和显式 error。旧 string-only API 仅在探针中适配为 `(reply,nil)`，不改旧代码行为。8/9 版本遗漏了返回瞬间取消，后续已修复。**这是九个选定场景的正确性回归，不是 RAG 准确率、真实模型表现、线上失败率统计或检索性能提升。** 40 题仍全部 pending_review，人工审定数为 0。

```console
python test/run_contract_comparison.py --revisions 9afe774372f2068eb7e57ecdf1c491965eace2e3 c3e62adde05d5413771b8b16dcf87e47ef6ccdfa bb0cdc16f8c99dc72b3257d666204ed8c6adafb5 --output-dir contract-results
```

命令只使用本地 Git 快照、缓存 Go 依赖和内存响应；旧版本按新契约预期失败，工具仍要求九个用例结果齐全。完整提交历史随交付包保留。最终补充回归还覆盖分发前取消的 error/done 事件，以及文件交付阶段取消时的 interrupted 任务状态、无最终答案和无落盘文件。

连接器发布使用汇总提交，原始本地提交 ID 可能不在 main 祖先中。仓库附带 `test/results/stage1_contract/revisions.bundle`，保存上述固定版本及最终代码回归版本 `8bb636d80f61d7649293c766353babea95e74bab`；以基线 `9afe774` 为前置提交。首次从 GitHub 克隆后先导入历史，再运行对照命令（均为本地 Git 操作）：

```console
git bundle verify test/results/stage1_contract/revisions.bundle
git fetch test/results/stage1_contract/revisions.bundle HEAD:refs/remotes/contract-fixture/final
```

最终本地回归：排除需要本地端口的 `internal/infrastructure/tool` 后，47 个 Go 包命令退出 0（19 个包通过测试，28 个包无测试，215 个测试/子测试通过）；MCP 包以 `-run '^$'` 编译通过，4 个 MCP socket 测试在受限环境未执行。25 项 Python 指标/审阅测试通过。前端没有变更，本轮未执行前端构建。仓库本轮未配置 GitHub Actions 工作流，不能将本地通过声称为远端 CI 通过。
