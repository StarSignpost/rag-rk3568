#include "rag/doc_loader.h"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace rag {

std::string DocLoader::read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

int DocLoader::detect_section(const std::string& line, int current_id) {
    // Match: "一、", "二、", ..., "十、"
    static const char* SECTION_NUMBERS[] = {
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"
    };
    static const int NUM_COUNT = sizeof(SECTION_NUMBERS) / sizeof(SECTION_NUMBERS[0]);

    for (int i = 0; i < NUM_COUNT; ++i) {
        std::string prefix = std::string(SECTION_NUMBERS[i]) + "、";
        if (line.find(prefix) == 0) {
            return i;  // section_id = 0-based index
        }
    }

    // Also match "十一、" pattern for completeness
    for (int i = 10; i < 20; ++i) {
        std::string prefix;
        if (i == 10) prefix = "十、";
        else if (i == 11) prefix = "十一、";
        else if (i == 12) prefix = "十二、";
        else if (i == 13) prefix = "十三、";
        else break;

        if (line.find(prefix) == 0) return i;
    }

    return current_id;  // not a section header
}

void DocLoader::chunk_section(const std::string& text, int section_id,
                               const std::string& title, int offset,
                               std::vector<Chunk>& out) {
    if (text.empty()) return;

    // Trim leading/trailing whitespace
    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\n' || text[start] == '\r'))
        start++;
    size_t end = text.size();
    while (end > start && (text[end-1] == ' ' || text[end-1] == '\n' || text[end-1] == '\r'))
        end--;

    std::string clean = text.substr(start, end - start);
    if ((int)clean.size() < config_.min_chunk) return;

    int pos = 0;
    while (pos < (int)clean.size()) {
        int remain = (int)clean.size() - pos;
        if (remain < config_.min_chunk) break;

        int chunk_len = std::min(config_.chunk_size, remain);

        // Don't cut in the middle of a UTF-8 character
        while (chunk_len > 0 && pos + chunk_len < (int)clean.size()) {
            unsigned char c = (unsigned char)clean[pos + chunk_len - 1];
            if ((c & 0x80) == 0 || (c & 0xC0) == 0xC0) break;
            chunk_len--;
        }

        Chunk c;
        c.id            = (int)out.size();
        c.text          = clean.substr(pos, chunk_len);
        c.section_id    = section_id;
        c.section_title = title;
        c.start_pos     = offset + start + pos;
        out.push_back(c);

        pos += (config_.chunk_size - config_.overlap);

        // Re-align to UTF-8 boundary after overlap step
        while (pos > 0 && pos < (int)clean.size()) {
            unsigned char cb = (unsigned char)clean[pos];
            if ((cb & 0x80) == 0 || (cb & 0xC0) == 0xC0) break;
            pos++;
        }
    }
}

std::vector<Chunk> DocLoader::load(const std::string& file_path) {
    std::string raw = read_file(file_path);
    if (raw.empty()) return {};

    std::vector<Chunk> chunks;
    chunks.reserve(raw.size() / (config_.chunk_size - config_.overlap) + 4);

    std::istringstream stream(raw);
    std::string line;

    int         current_section_id = -1;
    std::string current_section_title;
    std::string current_section_text;
    int         section_start = 0;

    while (std::getline(stream, line)) {
        // Remove trailing \r
        if (!line.empty() && line.back() == '\r') line.pop_back();

        int sec = detect_section(line, current_section_id);
        if (sec != current_section_id || current_section_id == -1) {
            // Flush previous section
            if (!current_section_text.empty() && current_section_id >= 0) {
                chunk_section(current_section_text, current_section_id,
                              current_section_title, section_start, chunks);
            }
            current_section_id    = sec;
            current_section_title = line;
            current_section_text.clear();
            section_start = (int)stream.tellg();
        } else {
            if (!current_section_text.empty()) current_section_text += "\n";
            current_section_text += line;
        }
    }

    // Flush last section
    if (!current_section_text.empty() && current_section_id >= 0) {
        chunk_section(current_section_text, current_section_id,
                      current_section_title, section_start, chunks);
    }

    return chunks;
}

} // namespace rag
