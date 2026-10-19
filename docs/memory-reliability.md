# 长期记忆与 Outbox 故障边界

本轮基线：`60174892146c0b97847f3db42b688b5b9da39233`。
验证方式为离线 Go 单元测试、内存 HTTP transport 与 SQL mock；不需要模型、GPU 或数据库服务。

2026-10-19 本轮回归：10 个包的 104 个顶层测试通过（含子测试共 167 项），`cmd/server` 编译通过、无测试文件。运行使用 Go 1.26.5、`GOMAXPROCS=2`、`GOPROXY=off` 和 `-count=1 -p 1`。该数字是测试覆盖结果，不是产品效果或性能指标。

## 实际修复

1. **模型抽取 → 提交 → 缓存**：异步偏好和记忆抽取共用逐条检查入口。先检查身份、key/value 和拼接内容，再获取 embedding，最后提交 PostgreSQL 记忆及 Outbox 事务。失败不更新本路径的 LTM 或偏好缓存；exchange 来源不写用户偏好。移除了安全检查之前的 `SaveBatch`。模型偏好抽取错误向上传递；显式 mock 模式仍可使用规则提取。异步 LLM/embedding 调用使用 30 秒上下文。
2. **整合的持久化边界**：锁定完整数据库记录并校验用户和版本，在同一事务写回合并内容、向量、标签、替代关系、访问时间和事件。保留数据库中的安全字段与 embedding 来源，回滚不返回可应用的 change set。单条记忆也执行过期检查；衰减从最近提交的 `UpdatedAt` 计时，避免重复施加整段创建年龄。
3. **处理中断恢复**：领取时同时选择到期 pending 事件和租约已过期的 processing 事件，继续使用 `FOR UPDATE SKIP LOCKED`。每次领取生成新的随机 claim token，确认、重试、死信更新均要求 token 和 processing 状态匹配。worker 的批次上下文不超过租约，取消后停止处理，保留事件等待回收，不消耗失败次数。修复 repair 事件 ID 与 PostgreSQL UUID 列不兼容的问题。
4. **旧事件重放**：投影写入前读取 PostgreSQL 当前记录（包含 tombstone），拒绝错误的事件目标或身份，跳过过期版本。删除后的旧 upsert 不再仅因目标行不存在而复活记忆；孤儿修复事件不能删除 PostgreSQL 中仍活跃的行。写入内容取自权威记录，同版本不同 hash 报错。

## 可重复验证

整合时另外排除已隔离、已替代或已删除记忆；有向量的记录只有 embedding 来源一致才参与合并。Neo4j 不可用时返回错误，写操作消费结果以暴露延迟提交错误，避免 worker 确认未完成的写入。

```sh
go test -p 1 ./internal/application/memoryprojection \
  ./internal/infrastructure/persistence/memoryoutbox \
  ./internal/infrastructure/projection/... \
  ./internal/infrastructure/persistence/memorytx \
  ./internal/domain/memory/longterm \
  ./internal/infrastructure/llm ./internal/application/chat \
  ./internal/application/memoryreconcile ./cmd/server
```

低资源环境可设置 `GOMAXPROCS=2`；依赖已缓存时设置 `GOPROXY=off GOSUMDB=off` 保持离线。测试使用替身返回的固定向量，不执行实际推理。真实 PostgreSQL 锁竞争、worker 崩溃重启、Milvus/Neo4j 的服务端并发行为仍需要独立集成环境验证。

## 保留的限制

- 这是至少一次投递与最终一致性设计。claim token 防止旧 worker 修改队列状态；目标存储的读取和写入仍非跨服务原子 CAS。权威版本检查之后发生并发更新、驱动不响应取消等极端情况仍可能短暂写入旧投影，需要后续事件或对账收敛。不能称为 exactly-once。
- 同一目标的 worker 必须使用相同租约长度；尚无动态续租。应按业务负载调整批量大小与租约，避免慢批次反复回收。
- `user_preferences` 仍是独立的 best-effort 写入，接口不返回持久化错误且不接收 context；同步规则提取仍直接更新偏好缓存。本轮保证针对模型派生路径，不等于所有偏好和 LTM 的跨表原子提交，也没有引入统一抽取幂等键。
- `UpdatedAt` 作为衰减检查点；内容更新也会重置检查点。独立衰减时间字段、长期效果评估属于后续工作。
- 对账器目前只比较首个分页，已处理 repair key 仍会抑制相同 key 的再次入队。权威检查保护活跃行免受错误孤儿删除，但未重写完整分页对账。Neo4j 节点/边事件目前共用节点存储，本轮不宣称实现独立图边投影。
- 本轮没有测线上吞吐、延迟分位数、恢复时间或 RAG 效果增量。

## 简历项目经历建议稿

**KnowFlow｜Go 智能知识助手与 RAG 系统**

- 基于 Go 构建知识问答与多阶段任务执行系统，集成混合检索、RRF 融合、查询改写、重排与父子分块，支持 DAG 任务编排。
- 完善长期记忆写入链路，以 PostgreSQL 为权威存储，通过事务 Outbox 驱动向量和图投影；补齐租约过期回收、领取所有权校验、版本与 tombstone 检查，并以离线故障测试覆盖取消、重试、提交失败及旧事件重放。
- 规范模型失败传播及 RAG 评测入口，区分真实结果、离线 fixture 和待审阅证据标签，为向量、混合、混合加重排三组对照提供可复现配置与逐题记录。

历史指标处理：用户截图中的 **0.83** 保留为原有实验口径；本轮没有复测或重新定义它，也没有证明该数字失实。使用时应沿用截图中原本的指标名称、数据集和实验条件，不能擅自把它写成 Recall@5、准确率或新版本结果。新增 40 题为另一批待证据审阅数据，与历史结果分开记录。未提供简历原文件，本段为可粘贴草稿。
