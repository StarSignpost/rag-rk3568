#!/bin/bash
# download_deps.sh — 在 PC 端执行，下载所有 LubanCat 2 所需的运行时库和模型
# 使用方法: chmod +x download_deps.sh && ./download_deps.sh
set -e

OUTDIR="${1:-./lubancat2_deps}"
mkdir -p "$OUTDIR"/{runtime,models}
cd "$OUTDIR"

echo "============================================"
echo "  下载 LubanCat 2 RAG 依赖"
echo "  目标目录: $(pwd)"
echo "============================================"

# ---- 1. ONNX Runtime v1.26.0 (aarch64) ----
echo ""
echo "[1/5] 下载 ONNX Runtime (aarch64)..."
wget -nc https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-aarch64-1.26.0.tgz \
     -P runtime/
echo "  ✓ ONNX Runtime 下载完成"

# ---- 2. rknn-llm runtime (尝试) ----
echo ""
echo "[2/5] 下载 rknn-llm 预编译包..."
# 从 Rockchip 官方云盘或 GitHub releases 获取
# 预编译包包含 librkllmrt.so 和头文件
wget -nc https://github.com/airockchip/rknn-llm/releases/download/release-v1.2.3/rkllm-runtime-linux-aarch64.tar.gz \
     -P runtime/ 2>/dev/null || {
    echo "  ⚠ rknn-llm 预编译包下载失败（可能需要从 Rockchip 云盘手动下载）"
    echo "  手动下载地址: https://console.rock-chips.com (提取码: rkllm)"
}
echo "  ✓ rknn-llm 下载完成"

# ---- 3. bge-small-zh 模型 (HuggingFace) ----
echo ""
echo "[3/5] 下载 bge-small-zh 模型 (HuggingFace)..."
pip install huggingface_hub -q 2>/dev/null || true

python3 << 'PYEOF'
import os, json
from huggingface_hub import snapshot_download

model_dir = "models/bge-small-zh"
os.makedirs(model_dir, exist_ok=True)

# Download model files
snapshot_download("BAAI/bge-small-zh", local_dir=model_dir,
                  local_dir_use_symlinks=False,
                  ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model.*"])

# Export to ONNX
try:
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    # Mean pooling wrapper for ONNX export
    class MeanPoolingModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.base(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            # Mean pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
            pooled = pooled / attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            return pooled

    pooled_model = MeanPoolingModel(model)

    dummy = tokenizer("测试", return_tensors="pt")
    torch.onnx.export(
        pooled_model,
        (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids']),
        "models/bge-small-zh.onnx",
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['sentence_embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'token_type_ids': {0: 'batch', 1: 'seq'},
        },
        opset_version=14
    )
    print("  ✓ bge-small-zh.onnx 导出成功")
except Exception as e:
    print(f"  ⚠ ONNX 导出失败 (可能缺少 torch/transformers): {e}")
    print("  请手动执行: pip install transformers torch onnx")
PYEOF

# ---- 4. Qwen2.5-1.5B 模型 ----
echo ""
echo "[4/5] 下载 Qwen2.5-1.5B-Instruct..."
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os

model_dir = "models/Qwen2.5-1.5B-Instruct"
os.makedirs(model_dir, exist_ok=True)

snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=model_dir,
                  local_dir_use_symlinks=False,
                  ignore_patterns=["*.safetensors", "*.bin", "*.msgpack", "*.h5"])

# Copy tokenizer.json as vocab
import shutil
vocab_src = os.path.join(model_dir, "tokenizer.json")
vocab_dst = "models/qwen_vocab.json"
if os.path.exists(vocab_src):
    shutil.copy(vocab_src, vocab_dst)
    print("  ✓ qwen_vocab.json 已提取")
PYEOF

# ---- 5. rknn-llm 模型转换 (如果安装了 rkllm-toolkit) ----
echo ""
echo "[5/5] Qwen → rknn 量化转换..."
python3 << 'PYEOF'
import os
qwen_dir = "models/Qwen2.5-1.5B-Instruct"
output = "models/qwen2.5-1.5b-int4.rknn"

# Check if rkllm-toolkit is available
try:
    import rkllm
    from rkllm import RKLLM
    print("  正在转换 Qwen2.5-1.5B → int4.rknn (可能需要 10-30 分钟)...")
    # 实际转换代码取决于 rkllm-toolkit 版本
    # 详见: https://github.com/airockchip/rknn-llm
    print("  请手动运行 rkllm-toolkit 转换脚本")
except ImportError:
    print("  ⚠ rkllm-toolkit 未安装，跳过 rknn 转换")
    print("  安装方式: pip install rkllm-toolkit")
    print(f"  然后在 PC 上转换: rkllm convert --model {qwen_dir} --quant W4A16 --output {output}")
PYEOF

# ---- 完成 ----
echo ""
echo "============================================"
echo "  下载完成！"
echo "============================================"
echo ""
echo "目录结构:"
find . -maxdepth 3 -type f -exec ls -lh {} \; 2>/dev/null
echo ""
echo "下一步 — 拷贝到 LubanCat 2:"
echo "  scp -r runtime/* cat@lubancat2:/usr/local/lib/"
echo "  scp -r models/*  cat@lubancat2:/home/cat/rag/models/"
echo ""
echo "然后在板子上设置环境变量:"
echo "  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
