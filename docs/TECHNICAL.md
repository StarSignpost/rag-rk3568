# 技术文档 — rag-rk3568

## 1. 系统概述

`rag-rk3568` 是一个面向 RK3568 嵌入式平台的端侧 RAG（检索增强生成）问答系统。它在 LubanCat 2 单板计算机（4GB RAM, 4×Cortex-A55, 1 TOPs NPU）上实现完整的文档索引、混合检索、上下文构建和 LLM 推理流水线，无需网络连接。

**技术栈**：C++17 · CMake · ONNX Runtime · rknn-llm · Qwen2.5-1.5B INT4 · bge-small-zh

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        RAGPipeline                           │
│  query() / followup() — 顶层编排                              │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ DocLoader│ Tokenizer│ Embedding│ BM25Index│  LLMInference   │
│ 文档切块  │ 分词/BPE │ ONNX推理  │ 稀疏检索  │  rknn-llm 推理  │
├──────────┴──────────┼──────────┼──────────┴─────────────────┤
│   HybridRetriever    │PageManager│   ContextBuilder          │
│   两阶段混合检索       │ KV Cache │   Qwen 模板填充           │
└─────────────────────┴──────────┴───────────────────────────┘
```

### 2.1 子系统职责

| 模块 | 文件 | 职责 | 关键算法 |
|------|------|------|----------|
| `DocLoader` | `src/doc_loader.cpp` | 文档加载、章节检测、滑动窗口切块 | 中文章节正则匹配 + UTF-8 安全截断 |
| `Tokenizer` | `src/tokenizer.cpp` | 中文分词（unigram+bigram）+ Qwen BPE 编码 | 停用词过滤双字合成 + 贪心最长匹配 BPE |
| `BM25Index` | `src/bm25_index.cpp` | 倒排索引构建 + BM25 统计评分 | 饱和 TF + IDF + 文档长度归一化（k1=1.2, b=0.75） |
| `Embedding` | `src/embedding.cpp` | ONNX Runtime 嵌入推理（bge-small-zh） | 单线程推理 · 内存模式复用 · L2 归一化 |
| `HybridRetriever` | `src/hybrid_retriever.cpp` | 两阶段混合检索：BM25 粗排 → 稠密重排 | 加权融合打分 + 章节命中奖励 + 关键词密度奖励 |
| `ContextBuilder` | `src/context_builder.cpp` | Qwen2.5 chat template 构建 | 按章节聚合检索结果 · 上下文截断 |
| `PageManager` | `src/kv_cache.cpp` | 页面式 KV Cache 管理 | MurmurHash3 内容寻址 · 固定/可驱逐块 · 碰撞防御 |
| `LLMInference` | `src/llm_inference.cpp` | rknn-llm 推理封装 | 批量 Prefill（单 IOCTL）+ 逐 Token Decode · 退化检测 |
| `RAGPipeline` | `src/rag_pipeline.cpp` | 顶层编排 | 串联全部子系统 · 首问/追问双路径 |

---

## 3. 数据流详解

### 3.1 初始化流程

```
                  ┌──────────────────┐
                  │ RAGConfig        │
                  │ .doc_path        │
                  │ .onnx_model_path │
                  │ .rknn_model_path │
                  │ .vocab_path      │
                  └──────┬───────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    DocLoader.load   Tokenizer      Embedding.init
    ┌──────┐        .load_vocab     (ONNX Runtime)
    │ 读取  │        (BPE 词表)       ┌──────────┐
    │ 章节  │              │          │ env      │
    │ 检测  │              │          │ session  │
    │ 切块  │              │          │ mem_info │
    └──┬───┘              │          └────┬─────┘
       ▼                  ▼               ▼
    chunks[]       tokenized_chunks   embedding 可用?
       │                  │               │
       └──────────────────┼───────────────┘
                          ▼
               HybridRetriever.build_index
               ┌──────────┬──────────┐
               │ BM25 建倒排│ 全量预编码 │
               └──────────┴──────────┘
                          │
                          ▼
               LLMInference.init
               (rknn-llm 上下文创建)
                          │
                          ▼
               PageManager: 固定 System Prompt KV 块
                          │
                          ▼
                    initialized_ = true
```

### 3.2 查询流程

```
用户问题: "仪表盘警告灯亮了怎么办"
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: BM25 粗排 (HybridRetriever)                 │
│                                                     │
│ Tokenizer.tokenize("仪表盘警告灯亮了怎么办")           │
│ → ["仪","表","盘","仪表","表盘","警告","告灯"...]     │
│                                                     │
│ BM25Index.search(query_terms, top_k=20)              │
│ → [(chunk_3, 2.45), (chunk_7, 2.31), ...]           │
│                                                     │
│ 公式: Σ IDF(t) × TF(t,d)×(k₁+1) / [TF(t,d)+k₁×(1-b+b×|d|/avg)] │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: 稠密重排 (HybridRetriever)                   │
│                                                     │
│ Embedding.encode_single("仪表盘警告灯亮了怎么办")      │
│ → query_vec[768] L2 归一化                           │
│                                                     │
│ 对每个 BM25 候选块计算:                               │
│   final = 0.4×BM25 + 0.6×cosine + section_bonus     │
│           + density_bonus                            │
│                                                     │
│ section_bonus: 查询词命中章节标题 → +0.15             │
│ density_bonus: 关键词(警告/故障/DTC...)密度 → max+0.15│
└─────────────────────┬───────────────────────────────┘
                      ▼
                 Top-3 检索块
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ ContextBuilder.build()                              │
│                                                     │
│ <|im_start|>system                                   │
│ 你是汽车仪表盘维修助手。规则: ...                      │
│ <|im_end|>                                           │
│ <|im_start|>user                                     │
│ 参考文档:                                             │
│ [二、仪表盘警告灯]                                     │
│   警告灯颜色含义...故障等级分类...                       │
│                                                      │
│ 问题: 仪表盘警告灯亮了怎么办                            │
│ <|im_end|>                                           │
│ <|im_start|>assistant                                │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ LLMInference.generate()                             │
│                                                     │
│ Phase 1: PageManager.map_tokens_to_blocks()          │
│   编码完整 prompt → token IDs[512]                   │
│   每 16 tokens 一页 → 32 页                          │
│   对每页计算 MurmurHash3 + 内容精确匹配                │
│   System prompt 页命中 → 直接复用 (dirty=false)       │
│   新页 → 分配并标记 dirty=true                        │
│                                                     │
│ Phase 2: batched_prefill                            │
│   收集所有 dirty 页的 tokens                         │
│   单次 rknn_llm_forward(prefill=true) → NPU         │
│   所有新 KV 数据写入 DMA-BUF                          │
│                                                     │
│ Phase 3: Decode Loop                                │
│   for i in 0..max_new_tokens:                       │
│     output = rknn_llm_decode(last_token, kv_info)    │
│     if output == <|endoftext|> → break              │
│     result += tokenizer.decode(output)               │
│     退化检测: 连续3字符相同 → break                    │
└─────────────────────┬───────────────────────────────┘
                      ▼
                  完整回答
```

### 3.3 追问流程（与首问的关键差异）

1. 检索：对新问题独立执行两阶段检索（不依赖历史上下文）
2. ContextBuilder：`build_followup()` 将完整聊天历史注入 prompt
3. KV Cache：释放上一轮上下文页面 → 新页面的系统 prompt 部分命中缓存，只需 prefill 增量 tokens
4. 解码：同首问

---

## 4. 核心算法详解

### 4.1 BM25 检索公式

```
IDF(t) = ln[(N - df + 0.5) / (df + 0.5) + 1]

TF_{saturated}(t, d) = tf_raw × (k₁ + 1) / (tf_raw + k₁ × (1 - b + b × |d|/avg_dl))

Score(d, q) = Σ_{t∈q} IDF(t) × TF_{saturated}(t, d)
```

参数：`k₁ = 1.2`（控制 TF 饱和速度），`b = 0.75`（长度归一化强度）

### 4.2 混合检索融合公式

```
final_score = 0.4 × BM25_score_norm
            + 0.6 × cosine(query_vec, chunk_vec)
            + section_bonus      (0.15 if any query term hits section title)
            + density_bonus      (min(hits × 0.03, 0.15), keywords like 警告/故障/DTC)
```

权重设计依据：
- BM25 权重 0.4：保证精确关键词匹配不被语义匹配淹没
- 稠密权重 0.6：语义层面召回 BM25 漏掉的同义表述
- section_bonus：章节标题匹配说明该节与问题高度相关
- density_bonus：技术关键词密度高的块更可能包含核心信息

### 4.3 页面式 KV Cache 管理

**设计目标**：在 4GB RAM 约束下，最大化跨轮对话的 KV 复用率。

**页面规格**：
- `KV_PAGE_SIZE = 16` tokens/页
- `KV_MAX_PAGES = 32` 页（总计 512 tokens 容量）
- 每页通过 ION/DMA-BUF 分配 NPU 物理内存

**内容寻址哈希**：
```cpp
hash = MurmurHash3_finalizer(token_ids[0..15])
// 碰撞防御: hash 命中后还需 exact token match
if (hash_to_block_[hash].matches(tokens)) → 复用
else → 分配新块 (防碰撞)
```

**驱逐策略**：
1. 优先驱逐 `pinned=false ∧ ref_count=0` 的块
2. 无候选时遍历所有非固定块强行驱逐
3. System prompt 块（`pinned=true`）永不被驱逐

**追问优化**：`followup()` 先释放上一轮上下文块的引用计数，再执行新查询的块映射。System prompt 块的 hash 不变 → 被新映射再次命中 → 跳过 prefill。

### 4.4 中文分词策略

**BM25 分词**（`Tokenizer::tokenize`）：
- Unigram：所有非停用词单字
- Bigram：两个连续非停用词 → 例如"仪表盘" → ["仪","表","盘","仪表","表盘"]
- 停用词表：30+ 常见虚词/标点（的/了/是/在/和/吗/呢...）
- Bigram 要求原始文本中两字相邻（中间无停用词）

**LLM 编码**（`Tokenizer::encode`）：
- BPE 贪心最长匹配（max token length=16 bytes）
- 无词表时：UTF-8 字节级回退（用于单元测试）

### 4.5 退化检测

```cpp
// 在 decode loop 中，每生成 5+ tokens 后检查
if (result[rlen-1] == result[rlen-2] && result[rlen-2] == result[rlen-3]) {
    break;  // 连续重复 → 停止生成
}
```

---

## 5. 内存预算

| 组件 | 内存占用 | 说明 |
|------|----------|------|
| Qwen2.5-1.5B INT4 | ~1 GB | rknn-llm 运行时加载 |
| bge-small-zh (ONNX) | ~96 MB | 权重 + 推理临时内存 |
| 嵌入向量缓存 | ~24 KB | 768×4 bytes × N chunks |
| BM25 倒排索引 | ~1-5 MB | 取决于文档大小 |
| KV Cache DMA BUF | ~8 MB | 32页 × 16 tokens × 特征维度 |
| 程序开销 | ~10 MB | 堆栈 + 代码段 |
| **总计** | **~1.2 GB** | 在 4GB RAM 内安全 |

---

## 6. 构建与交叉编译

### 6.1 本地开发编译（x86_64, SDK stub）

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
```

- 无 ONNX Runtime 时：嵌入模块返回零向量，BM25-only 检索
- 无 rknn-llm 时：使用 stub 函数编译通过，可运行单元测试和检索逻辑

### 6.2 交叉编译（LubanCat 2, aarch64）

```bash
./scripts/build_lubancat2.sh
```

等价于：
```bash
cmake -B build_lubancat2 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DONNXRUNTIME_DIR=/usr/lib/aarch64-linux-gnu \
    -DRKNNLLM_DIR=/usr/lib \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build_lubancat2 -j4
aarch64-linux-gnu-strip build_lubancat2/rag_cli
```

编译优化：`-O2 -march=armv8-a`

---

## 7. 测试策略

| 测试 | 文件 | 覆盖范围 |
|------|------|----------|
| `test_bm25` | `tests/test_bm25.cpp` | BM25 索引构建 + IDF/TF 公式验证 + 检索结果排序 |
| `test_retriever` | `tests/test_retriever.cpp` | 两阶段混合检索端到端 · 融合公式验证 |
| `test_kv_cache` | `tests/test_kv_cache.cpp` | 页面分配/回收 · 哈希命中/碰撞 · 驱逐逻辑 · 固定块保护 |

CMake 注册：`add_test(NAME bm25 COMMAND test_bm25)` → `ctest` 一键运行。

---

## 8. 扩展点

- **多文档支持**：`DocLoader.load()` 可连续调用，追加到 `chunks_` 后重建索引
- **模型替换**：`LLMInference::Config` 和 `Embedding::Config` 隔离模型路径，换模型只需改配置
- **聊天持久化**：`history_` 存储 `vector<ChatTurn>`，可序列化到磁盘实现会话恢复
- **ONNX 无依赖回退**：宏 `RAG_HAS_ONNX` 在缺失 ONNX Runtime 时编译为 BM25-only 模式
