#include "rag/context_builder.h"

#include <algorithm>
#include <map>

namespace rag {

const char* ContextBuilder::system_prompt() {
    return "<|im_start|>system\n"
           "你是汽车仪表盘维修助手。根据以下用户手册内容回答问题。\n"
           "规则:\n"
           "1. 只基于提供的文档内容作答，不要编造信息\n"
           "2. 如果文档不足以回答，直接说\"手册中未找到相关信息\"\n"
           "3. 回答分步骤，每步不超过一句话\n"
           "<|im_end|>\n";
}

const char* ContextBuilder::system_prompt_end() {
    return "<|im_end|>\n";
}

std::string ContextBuilder::merge_context(const std::vector<RetrievedChunk>& retrieved) {
    // Group by section
    std::map<int, std::vector<const RetrievedChunk*>> by_section;
    for (auto& rc : retrieved) {
        // Infer section_id from the section title
        for (char c : rc.section_title) {
            if (c >= 0x30 && c <= 0x39) { // crude but works
                int sid = c - '0';
                by_section[sid].push_back(&rc);
                break;
            }
        }
        if (by_section.empty() || by_section.find(0) == by_section.end()) {
            by_section[0].push_back(&rc);
        }
    }

    if (by_section.empty()) {
        for (auto& rc : retrieved) {
            by_section[0].push_back(&rc);
        }
    }

    std::string ctx;
    for (auto& [sid, rcs] : by_section) {
        // Find a title
        if (!rcs.empty() && !rcs[0]->section_title.empty()) {
            ctx += "[" + rcs[0]->section_title + "]\n";
        }
        for (auto* rc : rcs) {
            ctx += rc->text + "\n";
        }
    }

    return trim_context(ctx);
}

std::string ContextBuilder::trim_context(const std::string& ctx) {
    if ((int)ctx.size() <= config_.max_context_chars) return ctx;

    std::string trimmed = ctx.substr(0, config_.max_context_chars);
    auto last_nl = trimmed.rfind('\n');
    if (last_nl != std::string::npos && last_nl > (size_t)(config_.max_context_chars / 2)) {
        trimmed.resize(last_nl);
    }
    return trimmed;
}

std::string ContextBuilder::build(const std::vector<RetrievedChunk>& retrieved,
                                   const std::string& user_query) {
    std::string prompt;
    prompt += system_prompt();  // includes <|im_end|>
    prompt += "<|im_start|>user\n";
    prompt += "参考文档:\n";
    prompt += merge_context(retrieved);
    prompt += "\n问题: " + user_query + "\n";
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

std::string ContextBuilder::build_followup(const std::vector<RetrievedChunk>& retrieved,
                                            const std::string& user_query,
                                            const std::vector<ChatTurn>& history) {
    std::string prompt;

    // System prompt
    prompt += system_prompt();

    // Chat history
    for (auto& turn : history) {
        prompt += "<|im_start|>user\n";
        prompt += turn.query;
        prompt += "\n<|im_end|>\n";
        prompt += "<|im_start|>assistant\n";
        prompt += turn.response;
        prompt += "\n<|im_end|>\n";
    }

    // Current query with new context
    prompt += "<|im_start|>user\n";
    prompt += "参考文档:\n";
    prompt += merge_context(retrieved);
    prompt += "\n问题: " + user_query + "\n";
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";

    return prompt;
}

} // namespace rag
