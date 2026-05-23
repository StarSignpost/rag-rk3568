#pragma once

#include "types.h"

#include <string>
#include <vector>
#include <regex>

namespace rag {

// Loads and chunks a plain-text automotive manual.
// Chapter-aware: detects section headers like "二、仪表盘警告灯"
class DocLoader {
public:
    struct Config {
        int chunk_size  = 200;  // chars per chunk
        int overlap     = 40;   // sliding window overlap
        int min_chunk   = 50;   // discard chunks shorter than this
    };

    DocLoader() = default;
    explicit DocLoader(const Config& cfg) : config_(cfg) {}

    std::vector<Chunk> load(const std::string& file_path);

private:
    std::string read_file(const std::string& path);
    int         detect_section(const std::string& line, int current_id);
    void        chunk_section(const std::string& text, int section_id,
                              const std::string& title, int offset,
                              std::vector<Chunk>& out);

    Config config_;
};

} // namespace rag
