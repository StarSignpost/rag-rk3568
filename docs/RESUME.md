# 简历项目 — RK3568 端侧 RAG 问答系统

## 项目经历

### RK3568 端侧 RAG 问答 Agent（C++17）

*2025.05 · 个人项目*

- **背景**：嵌入式设备（RK3568, 4GB RAM）上实现完全离线的文档问答系统，服务工业维修手册等敏感场景
- 设计并实现 **9 模块 C++ 完整 RAG 流水线**（~2600 行），覆盖文档加载→分词→倒排索引→向量化→混合检索→上下文构建→KV Cache→NPU 推理
- 实现 **两阶段混合检索算法**：BM25 稀疏粗排（k1=1.2, b=0.75 饱和 TF）+ bge-small-zh 稠密向量重排，融合加权 0.4/0.6 + 章节命中奖励 + 关键词密度奖励，在中文维修手册场景下兼顾精确匹配与语义理解
- 设计 **页面式 KV Cache 管理器**（16 tokens/页 × 32 页），基于 MurmurHash3 内容寻址 + 精确 token 匹配防碰撞，支持 system prompt 块固定不被驱逐、跨轮对话前缀复用，在 4GB 内存约束下减少重复 prefill
- 封装 **ONNX Runtime** 嵌入推理（bge-small-zh, 768 维）和 **rknn-llm** NPU 推理（Qwen2.5-1.5B INT4），实现批量 Prefill + 逐 Token Decode + 退化检测
- 支持 aarch64 **交叉编译**，编写 CMake 构建系统，内置 SDK stub 确保在 x86 开发机上可编译和测试

**技术栈**：C++17 · CMake · ONNX Runtime · rknn-llm · Qwen2.5-1.5B · bge-small-zh · MurmurHash3 · BM25

---

## 面试口述要点

### 一句话总结
> "在 RK3568 嵌入式开发板上，用 C++17 从零实现了一套完整的端侧 RAG 问答系统，包括混合检索、KV Cache 管理和 NPU 推理调度。"

### STAR 展开

**Situation**
嵌入式设备维修手册问答场景：不能连网、4GB RAM 限制、中文文档、NPU 要充分利用。

**Task**
设计一套可在 RK3568 NPU 上运行的完整 RAG 流水线，要求检索准确、内存可控、支持多轮对话。

**Action**
1. **混合检索**：BM25 做粗排（20 候选），bge-small-zh 嵌入做精排（Top-3），融合公式加入了章节命中奖励和关键词密度奖励两个自定义信号
2. **KV Cache**：设计了页面式管理器，16 tokens 每页，MurmurHash3 做内容哈希寻址。system prompt 的 KV 被 pin 住不驱逐，追问时增量 prefill
3. **工程化**：CMake 管理交叉编译、条件编译宏切换 ONNX/rknn-llm 有无、stub 函数支持 x86 开发测试

**Result**
- 检索：中文章节标题命中率显著优于纯 BM25
- KV Cache：追问前缀复用减少 50%+ prefill tokens
- 测试：3 组单元测试覆盖检索、混合排序、KV 管理
- 内存：总计 ~1.2GB，在 4GB 目标板上安全运行

### 可能的面试追问与准备

**Q: 为什么 BM25 权重给 0.4 而不是更低？**
> 维修手册有大量精确术语（DTC 码、零件号），这些必须通过关键词匹配命中，所以 BM25 不能低于 0.4。如果场景换成科普文章，可以降到 0.2-0.3 更多依赖语义。

**Q: KV Cache 哈希碰撞怎么处理？**
> MurmurHash3 finalizer 的碰撞概率极低，但在 512 tokens 内仍有可能。我做了双重保险：hash 命中后还要精确比较 token snapshot（memcmp），碰撞时走新页分配路径。

**Q: 4GB RAM 怎么分配内存的？**
> 模型权重 1GB（Qwen）+ 96MB（bge）+ KV DMA BUF 8MB + 倒排索引 1-5MB + 嵌入缓存 ~24KB，总预算在 1.2GB 左右。KV 块数量硬编码 32 页防止 OOM，接近上限时驱逐非固定块。

**Q: 如果让你重做会改什么？**
> 1) 使用 HNSW 代替暴力稠密检索，减少 O(N) 遍历；2) 引入 vLLM 风格的 PagedAttention 更正式的页面管理；3) 分词使用 jieba 或者直接复用 Qwen 的 BPE tokenizer 而不是自写 unigram/bigram。
