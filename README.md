# rag-rk3568

RK3568 端侧 RAG 问答 Agent。基于 LubanCat 2 开发板，使用 Qwen2.5-1.5B（INT4 量化）作为生成模型，bge-small-zh 作为嵌入模型，实现本地的文档检索增强生成。

## 架构

```
用户问题 ──► BM25 粗排 (top-20) ──► 稠密向量重排 (top-3) ──► 上下文构建 ──► rknn-llm 推理 ──► 回答
                              │                      │
                        bm25_index            embedding (ONNX)
                                                    │
                                              bge-small-zh
```

- **两阶段检索**：BM25 稀疏检索召回候选，稠密向量余弦相似度精排，融合公式 `0.4 * BM25 + 0.6 * cosine + section_bonus + density_bonus`
- **页面式 KV Cache**：每页 16 tokens，基于内容哈希的命中检测，支持跨轮对话前缀复用，节省推理延迟
- **交互式 CLI**：支持独立提问和追问，保持对话历史与 KV 页面引用

## 模块

| 模块 | 文件 | 职责 |
|------|------|------|
| `tokenizer` | `src/tokenizer.cpp` | 文本分词（中文按字符切分） |
| `doc_loader` | `src/doc_loader.cpp` | 文档加载与切块 |
| `bm25_index` | `src/bm25_index.cpp` | BM25 稀疏检索索引 |
| `embedding` | `src/embedding.cpp` | ONNX Runtime 推理 bge-small-zh，输出 768 维向量 |
| `hybrid_retriever` | `src/hybrid_retriever.cpp` | 两阶段混合检索 |
| `context_builder` | `src/context_builder.cpp` | 基于检索结果构建 LLM prompt |
| `kv_cache` | `src/kv_cache.cpp` | 页面式 KV Cache 管理，支持固定/可驱逐块 |
| `llm_inference` | `src/llm_inference.cpp` | rknn-llm 推理封装，批量 prefill + 逐 token decode |
| `rag_pipeline` | `src/rag_pipeline.cpp` | 顶层编排器，组合所有子系统 |

## 依赖

- **rknn-llm** — Rockchip NPU LLM 推理 SDK（Qwen2.5-1.5B INT4）
- **ONNX Runtime** — 嵌入模型推理（bge-small-zh）
- **spdlog** — 日志（header-only）
- **CMake >= 3.16** — 构建系统
- **aarch64-linux-gnu 工具链** — 交叉编译

## 模型准备

将模型文件放入 `models/` 目录：

```
models/
├── bge-small-zh.onnx          # ONNX Runtime 嵌入模型
├── qwen2.5-1.5b-int4.rknn     # rknn-llm 量化模型
└── qwen_vocab.json            # Qwen 词表
```

## 构建

```bash
# 交叉编译 (LubanCat 2 / aarch64)
./scripts/build_lubancat2.sh

# 本地编译（需要安装对应依赖）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
```

## 运行

```bash
# 基本用法
./rag_cli

# 指定自定义路径
./rag_cli data/my_doc.txt models/my_llm.rknn models/my_emb.onnx

# 部署到 LubanCat 2
scp build_lubancat2/rag_cli cat@lubancat2:/home/cat/rag/
```

启动后进入交互式问答：

```
> 什么是RK3568？
[检索中...]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RK3568 是瑞芯微推出的...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> /help   显示帮助
> /doc    显示文档状态
> /quit   退出
```

## 测试

```bash
cd build
cmake .. -DBUILD_TESTS=ON
cmake --build .
ctest
```

- `test_bm25` — BM25 索引与检索正确性
- `test_retriever` — 混合检索端到端
- `test_kv_cache` — KV Cache 页面分配与命中

## 目标平台

- **硬件**：LubanCat 2（RK3568, 4GB RAM）
- **系统**：Armbian (Ubuntu 22.04 aarch64)
- **模型**：Qwen2.5-1.5B INT4（~1GB 权重） + bge-small-zh（~96MB）

## License

MIT
