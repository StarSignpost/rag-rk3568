#pragma once

#include "types.h"

#include <string>
#include <vector>
#include <map>

namespace rag {

// Builds Qwen2.5 chat-template prompts from retrieved chunks.
// Uses <|im_start|>/<|im_end|> markers, trims context to fit window.
class ContextBuilder {
public:
    struct Config {
        int  max_context_chars = 600;  // hard trim for 3 chunks
        bool enable_followup   = false;
    };

    ContextBuilder() = default;
    explicit ContextBuilder(const Config& cfg) : config_(cfg) {}

    // Build full prompt for initial query.
    std::string build(const std::vector<RetrievedChunk>& retrieved,
                      const std::string& user_query);

    // Build follow-up prompt with chat history.
    std::string build_followup(const std::vector<RetrievedChunk>& retrieved,
                               const std::string& user_query,
                               const std::vector<ChatTurn>& history);

    // Just the system prompt, kept for KV Cache pinning.
    static const char* system_prompt();
    static const char* system_prompt_end();

private:
    std::string merge_context(const std::vector<RetrievedChunk>& retrieved);
    std::string trim_context(const std::string& ctx);

    Config config_;
};

} // namespace rag
