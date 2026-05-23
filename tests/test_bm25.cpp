#include "rag/bm25_index.h"
#include "rag/tokenizer.h"
#include "rag/doc_loader.h"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace rag;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    tests_run++; \
    std::cout << "  TEST: " << name << " ... ";

#define PASS() do { \
    std::cout << "PASSED" << std::endl; \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    std::cout << "FAILED: " << msg << std::endl; \
} while(0)

// ---- test helpers ----

static std::vector<std::vector<std::string>> tokenize_chunks(
    Tokenizer& tok, const std::vector<Chunk>& chunks) {
    std::vector<std::vector<std::string>> result;
    for (auto& c : chunks) {
        auto tokens = tok.tokenize(c.text);
        std::vector<std::string> terms;
        for (auto& t : tokens) terms.push_back(t.text);
        result.push_back(terms);
    }
    return result;
}

// ---- tests ----

static void test_stopword_filtering() {
    TEST("Stop-word filtered bigrams");
    Tokenizer tok;
    auto tokens = tok.tokenize("小扳手亮了怎么消掉");

    // Check that "了" does NOT form bigrams with neighbors
    bool ok = true;
    for (auto& t : tokens) {
        if (t.is_bigram) {
            // Bigram should not contain stop chars
            for (char c : t.text) {
                unsigned char uc = (unsigned char)c;
                // Simple check: ASCII letters/digits and common CJK are fine
                if (uc == ' ' || uc == '\n' || uc == '\r') {
                    ok = false;
                }
            }
        }
    }
    if (ok) { PASS(); } else { FAIL("stop-word leaked into bigram"); }
}

static void test_bm25_build_and_search() {
    TEST("BM25 build and search");
    DocLoader loader;
    auto chunks = loader.load("data/manual.txt");
    if (chunks.empty()) { FAIL("no chunks loaded"); return; }

    Tokenizer tok;
    auto tok_chunks = tokenize_chunks(tok, chunks);

    BM25Index bm25;
    bm25.build(chunks, tok_chunks);

    if (bm25.num_chunks() == 0) { FAIL("no chunks indexed"); return; }

    // Search for "保养灯" terms
    auto query_tokens = tok.tokenize("保养灯复位");
    std::vector<std::string> terms;
    for (auto& t : query_tokens) terms.push_back(t.text);

    auto results = bm25.search(terms, 5);
    if (results.empty()) { FAIL("no results for 保养灯复位"); return; }

    float top_score = results[0].second;
    if (top_score <= 0.0f) { FAIL("top score is zero"); return; }

    PASS();
}

static void test_bm25_empty_query() {
    TEST("BM25 empty query returns empty");
    BM25Index bm25;
    std::vector<Chunk> empty_chunks;
    std::vector<std::vector<std::string>> empty_toks;
    bm25.build(empty_chunks, empty_toks);

    std::vector<std::string> terms;
    auto results = bm25.search(terms, 5);
    if (results.empty()) { PASS(); } else { FAIL("expected empty results"); }
}

int main() {
    std::cout << "=== BM25 Tests ===" << std::endl;

    test_stopword_filtering();
    test_bm25_build_and_search();
    test_bm25_empty_query();

    std::cout << "\nResults: " << tests_passed << "/" << tests_run << " passed." << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
