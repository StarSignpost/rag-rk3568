#pragma once

#include "tokenizer.h"
#include "kv_cache.h"

#include <string>
#include <vector>
#include <memory>

// rknn-llm SDK types (opaque forward declarations)
struct rknn_llm_context;
struct rknn_llm_kv_cache_info;

namespace rag {

// rknn-llm inference wrapper for Qwen2.5-1.5B (INT4 quantized).
// Uses batched prefill (single IOCTL for all new tokens) and
// page-based KV cache management with content-addressable hashing.
class LLMInference {
public:
    struct Config {
        std::string model_path;       // Qwen2.5-1.5B-int4.rknn
        int         max_context   = 512;
        int         max_new_tokens = 128;
        float       temperature   = 0.0f;   // greedy decode for offline use
        int         eos_token_id  = 151643; // Qwen <|endoftext|>
    };

    LLMInference() = default;
    ~LLMInference();

    bool init(const Config& cfg, Tokenizer* tokenizer);
    bool is_ready() const { return ctx_ != nullptr; }

    // Full generation: prefill context tokens then decode.
    // tokens: the complete encoded prompt
    // page_mgr: page manager for KV cache block management
    std::string generate(const std::vector<int>& tokens, PageManager& page_mgr);

    // Decode-only step (for followup with full prefix reuse).
    int  decode_step(int input_id, const std::vector<int>& block_ids);

    const Config& config() const { return config_; }

private:
    void batched_prefill(const std::vector<int>& tokens,
                         const std::vector<int>& block_ids,
                         PageManager& page_mgr);

    Config             config_;
    rknn_llm_context*  ctx_      = nullptr;
    Tokenizer*         tokenizer_ = nullptr;
};

} // namespace rag
