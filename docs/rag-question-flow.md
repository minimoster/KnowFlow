# 知识库问答链路的失败语义

本轮范围是 `ProcessContext` / `ProcessStream` 中开启 `UseRAG` 的简单问答：
路由 → 检索 → PG 证据回填 → 答案生成 → 响应/历史写入。

| 情况 | 行为 |
| --- | --- |
| 开启知识库，但尚未加载文档 | 保持 `rag` 路由，提示先上传文档，不转入普通 Agent |
| 检索基础设施不可用 | 返回错误，不描述为“知识库为空” |
| 查询成功、没有命中或没有非空证据 | 返回“知识库中未找到相关内容”，不调用答案生成模型 |
| 有有效证据 | 原有父块回填、改写、重排机制继续生效，再调用模型 |
| 向量化/ES/Milvus/PG 读取失败 | 返回错误，不静默切换检索分支或把部分结果当成功 |
| ES 超时、分片失败、HTTP 错误；PG 行扫描/迭代失败；索引引用缺失的 PG 行 | 返回错误，不生成答案 |
| 一个改写查询失败 | 整轮失败，不使用其他查询的部分证据继续生成 |
| 生成报错、空答案、请求取消或超时 | 返回错误；迟到的生成结果不会发送为成功 token |

流式成功事件为 `route → rag_result → token → done`；检索/生成失败为
`route → error → done`。失败响应不进入 `finalize`，不写 assistant 历史或启动答案记忆提取。
已接收的 user 历史和独立的输入偏好提取仍按现有规则执行。
正常的无证据提示仍作为一轮回复留在对话历史中。

请求 context 现在传入 RAG 仓储的三个读方法（`SearchES`、`SearchMilvus`、
`LoadByIDs`），分别进入 ES HTTP 请求、Milvus SDK 和 `QueryContext`。
仓储接口的实现者需要更新这三个方法签名。摄入、启动初始化和删除 API 未改。
启动时自动选择 keyword/semantic/hybrid 模式仍保留；本轮改变的是已选模式在请求期间
发生错误时的处理，原来的静默运行时降级改为显式失败。

离线复现（不启动服务，不使用真实模型）：

```sh
GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off GOMAXPROCS=2 \
go test -count=1 -p 1 ./internal/domain/rag \
  ./internal/infrastructure/persistence/ragchunk \
  ./internal/application/chat ./internal/infrastructure/llm
```

测试使用内存 HTTP transport、SQLmock、Milvus 接口替身和固定证据。聊天集成测试
调用真实 `ProcessContext` / `ProcessStream`、RAG engine、生产生成回调及真实模式 LLM
客户端；覆盖三种检索模式、空库/空结果/空白证据、依赖错误、缺失行、429、空答案、
取消、迟到结果，以及已登录用户失败时不产生 assistant 记忆。仓储测试另检验 ES
错误/不完整响应、取消、SQL 扫描/迭代错误与超时、Milvus 逐查询错误。

边界：这是流程正确性测试，不证明召回率、答案质量或生产服务的性能。已有历史评测数字
保持原有归属，未产生新的真实 RAG 提升结论。“无证据”仅指没有可用检索文本，
不等于已经实现语义充分性判断或引用核验。可选 KG 的旧实体抽取/图检索 API、
报告 `rag_agent`、记忆检索前缀与异步记忆任务不在本轮取消保证范围内；这些路径仍需
后续专项处理。启动恢复仍用旧的 `LoadAll` / `Loaded` 标志：连接对象存在但启动读取
失败时，目前仍可能被显示为空库；本轮的读取错误保证针对请求内的检索和证据回填，
不包含启动状态判定。本轮未运行真实 ES/Milvus/PG/Neo4j 集成、GPU 或付费模型。
