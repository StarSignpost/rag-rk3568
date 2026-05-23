#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rag {

struct Chunk {
    int         id;
    std::string text;
    int         section_id;
    std::string section_title;
    int         start_pos;

    bool        section_title_matched = false;
};

struct RAGConfig {
    std::string doc_path;
    std::string onnx_model_path;
    std::string rknn_model_path;
    std::string vocab_path;
    int         max_context     = 512;
    int         max_new_tokens  = 128;
};

struct RetrievedChunk {
    int         chunk_id;
    float       score;
    std::string text;
    std::string section_title;
};

struct ChatTurn {
    std::string           query;
    std::string           response;
    std::vector<int>      response_token_ids;
    std::vector<int>      context_page_ids;
};

} // namespace rag
