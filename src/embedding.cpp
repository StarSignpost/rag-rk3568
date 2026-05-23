#include "rag/embedding.h"

#ifdef RAG_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#else
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#endif

#include <algorithm>
#include <cmath>

namespace rag {

Embedding::~Embedding() {
#ifdef RAG_HAS_ONNX
    delete static_cast<Ort::MemoryInfo*>(mem_info_);
    delete static_cast<Ort::Session*>(session_);
    delete static_cast<Ort::Env*>(env_);
#endif
}

bool Embedding::init(const Config& cfg) {
    cfg_ = cfg;

#ifdef RAG_HAS_ONNX
    try {
        Ort::Env* e = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "rag_embed");
        env_ = e;

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts.EnableMemPattern();                    // cross-layer memory reuse
        opts.DisableCpuMemArena();                  // no arena allocator on aarch64
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session* s = new Ort::Session(*e, cfg.model_path.c_str(), opts);
        session_ = s;

        Ort::MemoryInfo* m = new Ort::MemoryInfo(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        mem_info_ = m;

        return true;
    } catch (const std::exception& ex) {
        return false;
    }
#else
    // No ONNX Runtime available — embeddings must be pre-computed or
    // provided via an external file. Return false to signal caller.
    (void)cfg;
    return false;
#endif
}

std::vector<float> Embedding::encode_single(const std::string& text) {
    std::vector<float> result(cfg_.embedding_dim, 0.0f);

#ifdef RAG_HAS_ONNX
    try {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids;

        if (!build_input(text, input_ids, attention_mask, token_type_ids)) {
            return result;
        }

        int seq_len = (int)input_ids.size();
        std::vector<int64_t> shape = {1, seq_len};

        Ort::Env*        e = static_cast<Ort::Env*>(env_);
        Ort::Session*    s = static_cast<Ort::Session*>(session_);
        Ort::MemoryInfo* m = static_cast<Ort::MemoryInfo*>(mem_info_);

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            *m, input_ids.data(), input_ids.size(), shape.data(), shape.size());
        Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
            *m, attention_mask.data(), attention_mask.size(), shape.data(), shape.size());
        Ort::Value type_tensor = Ort::Value::CreateTensor<int64_t>(
            *m, token_type_ids.data(), token_type_ids.size(), shape.data(), shape.size());

        const char* input_names[]  = {"input_ids", "attention_mask", "token_type_ids"};
        const char* output_names[] = {"sentence_embedding"};

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        inputs.push_back(std::move(mask_tensor));
        inputs.push_back(std::move(type_tensor));

        auto outputs = s->Run(Ort::RunOptions{nullptr},
                              input_names, inputs.data(), inputs.size(),
                              output_names, 1);

        if (!outputs.empty() && outputs[0].IsTensor()) {
            float* data = outputs[0].GetTensorMutableData<float>();
            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            size_t count = info.GetElementCount();
            result.assign(data, data + std::min(count, (size_t)cfg_.embedding_dim));
        }
    } catch (...) {
        // Return zero vector on failure
    }
#else
    (void)text;
#endif

    return result;
}

bool Embedding::build_input(const std::string& text,
                             std::vector<int64_t>& input_ids,
                             std::vector<int64_t>& attention_mask,
                             std::vector<int64_t>& token_type_ids) {
    input_ids.clear();
    attention_mask.clear();
    token_type_ids.clear();

    // Simple UTF-8 tokenization: each Chinese character → one ID
    // For bge-small-zh, we'd need the actual tokenizer.
    // This is a fallback that maps bytes to the model's vocabulary range.
    // In production, use the actual BGE tokenizer from the model's tokenizer.json.

    // Add [CLS] token (ID 101 for BERT-based models)
    input_ids.push_back(101);
    attention_mask.push_back(1);
    token_type_ids.push_back(0);

    // Encode each UTF-8 character
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        int len = 1;
        if ((c & 0x80) == 0)       len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { i++; continue; }

        if (i + len > text.size()) break;

        // Map to a token ID in the model's range (100-21127 for BERT vocab)
        // Real implementation would use the tokenizer from the model
        int token_id = 100 + ((int)c % 21000);
        input_ids.push_back(token_id);
        attention_mask.push_back(1);
        token_type_ids.push_back(0);

        i += len;
    }

    // Add [SEP] token (ID 102)
    input_ids.push_back(102);
    attention_mask.push_back(1);
    token_type_ids.push_back(0);

    // Truncate to max_seq_len
    if ((int)input_ids.size() > cfg_.max_seq_len) {
        input_ids.resize(cfg_.max_seq_len);
        attention_mask.resize(cfg_.max_seq_len);
        token_type_ids.resize(cfg_.max_seq_len);
        input_ids.back() = 102;  // ensure [SEP] at end
    }

    return !input_ids.empty();
}

} // namespace rag
