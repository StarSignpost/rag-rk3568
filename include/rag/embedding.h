#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cmath>

// Forward declare ORT types (avoid pulling full ONNX Runtime headers here)
struct OrtEnv;
struct OrtSession;
struct OrtMemoryInfo;
struct OrtValue;

namespace rag {

// ONNX Runtime wrapper for bge-small-zh embedding model.
// Hard-coded to single-thread, memory-pattern reuse for aarch64.
// Peak memory: ~96MB weights + ~8MB temporary per forward pass.
class Embedding {
public:
    struct Config {
        std::string model_path;
        int         embedding_dim = 768;  // bge-small-zh
        int         max_seq_len   = 256;
    };

    Embedding() = default;
    ~Embedding();

    bool init(const Config& cfg);
    bool is_ready() const { return session_ != nullptr; }

    // Encode a single text to a float vector [embedding_dim].
    // Thread-safe to call sequentially. Peak allocation ~8MB.
    std::vector<float> encode_single(const std::string& text);

    int embedding_dim() const { return cfg_.embedding_dim; }

private:
    bool build_input(const std::string& text,
                     std::vector<int64_t>& input_ids,
                     std::vector<int64_t>& attention_mask,
                     std::vector<int64_t>& token_type_ids);

    Config cfg_;

    OrtEnv*        env_      = nullptr;
    OrtSession*    session_  = nullptr;
    OrtMemoryInfo* mem_info_ = nullptr;
};

// L2 normalization in-place.
inline void l2_normalize(std::vector<float>& v) {
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    float norm = std::sqrt(sum);
    if (norm < 1e-8f) return;
    for (float& x : v) x /= norm;
}

// Dot product of two same-length vectors.
inline float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

} // namespace rag
