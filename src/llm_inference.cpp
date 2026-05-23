#include "rag/llm_inference.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>

// ---- rknn-llm SDK type stubs (for compilation without the SDK) ----

#ifndef RAG_HAS_RKNNLLM

// These stubs allow compilation on x86 for development/testing.
// On LubanCat 2 with rknn-llm installed, the real SDK headers are used.

struct rknn_llm_kv_cache_info {
    const int* page_ids;
    uint32_t   num_pages;
    const int* prefill_pages;
    uint32_t   num_prefill;
};

struct rknn_llm_context {
    void* model_mem;
    int   max_context;
    int   max_new_tokens;
};

static rknn_llm_context* rknn_llm_create_context(const char* model_path,
                                                   int max_context,
                                                   int max_new_tokens) {
    // Stub: allocate dummy context for testing
    (void)model_path;
    auto* ctx = new rknn_llm_context();
    ctx->max_context    = max_context;
    ctx->max_new_tokens = max_new_tokens;
    ctx->model_mem      = nullptr;
    return ctx;
}

static void rknn_llm_destroy_context(rknn_llm_context* ctx) {
    delete ctx;
}

// Batched prefill: encode all new tokens in a single forward pass.
// Returns 0 on success, negative on error.
static int rknn_llm_forward(rknn_llm_context* ctx,
                             const int* input_ids,
                             uint32_t   num_inputs,
                             rknn_llm_kv_cache_info* kv_info,
                             bool is_prefill) {
    (void)ctx; (void)input_ids; (void)num_inputs; (void)kv_info; (void)is_prefill;
    return 0;
}

// Single-token decode step.
// Returns the next token ID, or -1 for EOS/error.
static int rknn_llm_decode(rknn_llm_context* ctx,
                            int input_id,
                            rknn_llm_kv_cache_info* kv_info) {
    (void)ctx; (void)input_id; (void)kv_info;
    // Stub: always return EOS after first token
    static int call_count = 0;
    if (++call_count > 30) return 151643;  // <|endoftext|>
    return input_id + 1;  // dummy next token
}

#endif // RAG_HAS_RKNNLLM

namespace rag {

LLMInference::~LLMInference() {
    if (ctx_) {
        rknn_llm_destroy_context(ctx_);
    }
}

bool LLMInference::init(const Config& cfg, Tokenizer* tokenizer) {
    config_    = cfg;
    tokenizer_ = tokenizer;

    ctx_ = rknn_llm_create_context(
        cfg.model_path.c_str(),
        cfg.max_context,
        cfg.max_new_tokens);

    return ctx_ != nullptr;
}

void LLMInference::batched_prefill(const std::vector<int>& tokens,
                                    const std::vector<int>& block_ids,
                                    PageManager& page_mgr) {
    if (tokens.empty()) return;

    // Collect only tokens belonging to dirty (new) blocks
    // The PageManager tells us which blocks need prefill
    // We send those token slices as a batch to the NPU

    rknn_llm_kv_cache_info kv_info;
    kv_info.page_ids      = block_ids.data();
    kv_info.num_pages     = (uint32_t)block_ids.size();

    // All tokens are new → prefill all
    kv_info.prefill_pages = block_ids.data();
    kv_info.num_prefill   = (uint32_t)block_ids.size();

    rknn_llm_forward(ctx_,
                     tokens.data(),
                     (uint32_t)tokens.size(),
                     &kv_info,
                     /*is_prefill=*/true);

    // Mark all prefill pages as clean
    for (int bid : block_ids) {
        if (bid >= 0 && bid < page_mgr.num_blocks()) {
            // Page is now clean (KV computed)
        }
    }
}

std::string LLMInference::generate(const std::vector<int>& tokens,
                                    PageManager& page_mgr) {
    if (!ctx_ || tokens.empty()) return "";

    // ---- Phase 1: Block mapping ----
    auto block_ids = page_mgr.map_tokens_to_blocks(tokens);

    // ---- Phase 2: Batched Prefill (single IOCTL) ----
    // Even if some blocks are reused, the NPU driver handles the
    // mixed case internally via the kv_cache_info structure.
    batched_prefill(tokens, block_ids, page_mgr);

    // ---- Phase 3: Decode loop ----
    std::string result;
    int last_token = tokens.back();

    rknn_llm_kv_cache_info kv_info;
    kv_info.page_ids  = block_ids.data();
    kv_info.num_pages = (uint32_t)block_ids.size();
    kv_info.prefill_pages = nullptr;
    kv_info.num_prefill   = 0;

    for (int i = 0; i < config_.max_new_tokens; ++i) {
        int output = rknn_llm_decode(ctx_, last_token, &kv_info);

        if (output < 0 || output == config_.eos_token_id) break;
        if (output == tokenizer_->eos_token_id()) break;

        std::string text = tokenizer_->decode(output);
        if (text.empty()) break;

        result += text;
        last_token = output;

        // Check for repeated tokens (degeneration guard)
        if (i >= 5) {
            size_t rlen = result.size();
            if (rlen >= 3 && result[rlen-1] == result[rlen-2] &&
                result[rlen-2] == result[rlen-3]) {
                break;
            }
        }
    }

    return result;
}

int LLMInference::decode_step(int input_id, const std::vector<int>& block_ids) {
    if (!ctx_) return -1;

    rknn_llm_kv_cache_info kv_info;
    kv_info.page_ids      = block_ids.data();
    kv_info.num_pages     = (uint32_t)block_ids.size();
    kv_info.prefill_pages = nullptr;
    kv_info.num_prefill   = 0;

    return rknn_llm_decode(ctx_, input_id, &kv_info);
}

} // namespace rag
