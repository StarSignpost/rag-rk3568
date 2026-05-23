#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace rag {

// Hybrid Unigram + Stop-word Filtered Bigram tokenizer.
// Produces both single-char unigrams and bigrams where neither char is
// a stop word. Designed for Chinese automotive manual search.
class Tokenizer {
public:
    Tokenizer();

    struct Token {
        std::string text;   // "小扳" or "消"
        bool        is_bigram;  // true = bigram, false = unigram
    };

    // Tokenize raw text into unigrams + filtered bigrams.
    std::vector<Token> tokenize(const std::string& text) const;

    // Qwen-compatible BPE tokenizer (tiktoken-like).
    // Returns token IDs for feeding into rknn-llm.
    bool load_vocab(const std::string& path);
    std::vector<int>  encode(const std::string& text) const;
    std::string       decode(int token_id) const;
    std::string       decode(const std::vector<int>& ids) const;
    int               vocab_size() const;
    int               eos_token_id() const;

private:
    bool is_stop_char(char c) const;
    int  find_token_id(const std::string& token) const;

    // 30+ stop chars — hardcoded lookup, O(1)
    bool stop_table_[256] = {false};

    // Qwen vocabulary
    std::vector<std::string> vocab_;       // id → text
    std::unordered_map<std::string, int>  token_to_id_;
    int eos_id_ = -1;
};

} // namespace rag
