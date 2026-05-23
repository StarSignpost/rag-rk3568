#include "rag/hybrid_retriever.h"
#include "rag/embedding.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace rag {

void HybridRetriever::build_index(const std::vector<Chunk>& chunks,
                                   const std::vector<std::vector<std::string>>& tokenized_chunks,
                                   Embedding* embedding) {
    chunks_    = chunks;
    embedding_ = embedding;

    // Build BM25 index
    bm25_ = std::make_unique<BM25Index>(1.2f, 0.75f);
    bm25_->build(chunks, tokenized_chunks);

    // Offline embedding: encode all chunks, L2-normalize
    chunk_embeddings_.reserve(chunks.size());
    int dim = embedding_->embedding_dim();

    for (const auto& c : chunks) {
        auto emb = embedding_->encode_single(c.text);
        if (emb.empty()) {
            emb.resize(dim, 0.0f);
        }
        l2_normalize(emb);
        chunk_embeddings_.push_back(std::move(emb));
    }
}

float HybridRetriever::density_bonus(const Chunk& chunk) const {
    static const std::vector<std::string> keywords = {
        "\xe8\xad\xa6\xe5\x91\x8a",     // 警告
        "\xe6\xb3\xa8\xe6\x84\x8f",     // 注意
        "\xe6\x95\x85\xe9\x9a\x9c",     // 故障
        "DTC",
        "\xe5\xa4\x8d\xe4\xbd\x8d",     // 复位
        "\xe4\xbf\x9d\xe5\x85\xbb",     // 保养
        "\xe5\xbc\x82\xe5\xb8\xb8",     // 异常
        "\xe6\xa3\x80\xe6\x9f\xa5",     // 检查
    };

    int hits = 0;
    for (const auto& kw : keywords) {
        size_t p = 0;
        while ((p = chunk.text.find(kw, p)) != std::string::npos) {
            hits++;
            p += kw.size();
        }
    }

    return std::min(hits * 0.03f, 0.15f);
}

void HybridRetriever::compute_section_hits(const std::vector<std::string>& query_terms) {
    section_hit_.clear();

    for (const auto& term : query_terms) {
        for (const auto& chunk : chunks_) {
            if (chunk.section_title.find(term) != std::string::npos) {
                section_hit_[chunk.section_id] = true;
            }
        }
    }
}

float HybridRetriever::rerank_score(int chunk_id, const std::vector<float>& query_vec) {
    float bm25_score   = bm25_scores_.count(chunk_id) ? bm25_scores_[chunk_id] : 0.0f;
    float dense_score  = dot_product(query_vec, chunk_embeddings_[chunk_id]);
    float section_bonus = section_hit_.count(chunks_[chunk_id].section_id) ? 0.15f : 0.0f;
    float dens_bonus    = density_bonus(chunks_[chunk_id]);

    return 0.4f * bm25_score + 0.6f * dense_score + section_bonus + dens_bonus;
}

std::vector<RetrievedChunk> HybridRetriever::retrieve(const std::string& query) {
    if (!bm25_ || !embedding_) return {};

    // ---- Stage 1: Tokenize query + BM25 coarse ----
    auto tokens = tokenizer_.tokenize(query);

    std::vector<std::string> query_terms;
    query_terms.reserve(tokens.size());
    for (const auto& t : tokens) {
        query_terms.push_back(t.text);
    }

    auto bm25_results = bm25_->search(query_terms, config_.bm25_top_k);

    // Store BM25 scores for re-ranking
    bm25_scores_.clear();
    for (auto& [chunk_id, score] : bm25_results) {
        bm25_scores_[chunk_id] = score;
    }

    // Compute section title hits
    compute_section_hits(query_terms);

    // ---- Stage 2: Encode query + dense re-rank ----
    auto query_vec = embedding_->encode_single(query);
    if (query_vec.empty()) {
        query_vec.resize(embedding_->embedding_dim(), 0.0f);
    }
    l2_normalize(query_vec);

    // Re-rank
    std::vector<std::pair<int, float>> scored;
    scored.reserve(bm25_results.size());

    for (auto& [chunk_id, _] : bm25_results) {
        float fs = rerank_score(chunk_id, query_vec);
        scored.push_back({chunk_id, fs});
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    int top_n = std::min(config_.final_top_n, (int)scored.size());

    std::vector<RetrievedChunk> result;
    result.reserve(top_n);
    for (int i = 0; i < top_n; ++i) {
        int chunk_id = scored[i].first;
        RetrievedChunk rc;
        rc.chunk_id      = chunk_id;
        rc.score         = scored[i].second;
        rc.text          = chunks_[chunk_id].text;
        rc.section_title = chunks_[chunk_id].section_title;
        result.push_back(rc);
    }

    return result;
}

} // namespace rag
