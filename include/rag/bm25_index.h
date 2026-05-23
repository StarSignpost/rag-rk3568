#pragma once

#include "types.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace rag {

// Pure BM25 text statistics engine.
// Does NOT access Chunk objects — only word frequencies.
class BM25Index {
public:
    struct TermMatch {
        std::string        text;
        int                df;          // document frequency
        std::vector<float> tf;          // term frequency per chunk, indexed by chunk_id
    };

    BM25Index(float k1 = 1.2f, float b = 0.75f)
        : k1_(k1), b_(b) {}

    void build(const std::vector<Chunk>& chunks,
               const std::vector<std::vector<std::string>>& tokenized_chunks);

    // Return BM25 score for chunk_id given query term matches.
    // Pure statistical: no section title, no chunk content access.
    float score(const std::vector<TermMatch>& matches, int chunk_id) const;

    // Search: tokenize query, build TermMatch list, return chunk_id→score map.
    std::vector<std::pair<int, float>> search(
        const std::vector<std::string>& query_terms, int top_k) const;

    int num_chunks() const { return N_; }

private:
    float avg_len_;
    int   N_ = 0;
    float k1_;
    float b_;
    std::vector<float> doc_len_norm_;

    // Inverted index: term → list of (chunk_id, tf)
    std::unordered_map<std::string, std::vector<std::pair<int, float>>> inverted_;
};

} // namespace rag
