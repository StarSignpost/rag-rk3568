#include "rag/rag_pipeline.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════╗
║      RK3568 端侧 RAG 问答 Agent              ║
║      LubanCat 2 · Qwen2.5-1.5B · bge-small  ║
╚══════════════════════════════════════════════╝
)" << std::endl;
}

static void print_usage() {
    std::cout << "Commands:\n"
              << "  /help     Show this help\n"
              << "  /quit     Exit\n"
              << "  /doc      Show loaded document stats\n"
              << "  Any other input is treated as a question.\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    print_banner();

    // ---- Configuration ----
    rag::RAGConfig cfg;
    cfg.doc_path        = "data/manual.txt";
    cfg.onnx_model_path = "models/bge-small-zh.onnx";
    cfg.rknn_model_path = "models/qwen2.5-1.5b-int4.rknn";
    cfg.vocab_path      = "models/qwen_vocab.json";
    cfg.max_context     = 512;
    cfg.max_new_tokens  = 128;

    // Allow overriding paths from command line
    if (argc > 1) cfg.doc_path = argv[1];
    if (argc > 2) cfg.rknn_model_path = argv[2];
    if (argc > 3) cfg.onnx_model_path = argv[3];

    // ---- Initialize ----
    rag::RAGPipeline pipeline;
    if (!pipeline.initialize(cfg)) {
        std::cerr << "Failed to initialize RAG pipeline. Check:\n"
                  << "  - " << cfg.doc_path << " exists\n"
                  << "  - " << cfg.rknn_model_path << " exists\n"
                  << "  - " << cfg.onnx_model_path << " exists (optional)\n"
                  << "  - " << cfg.vocab_path << " exists (optional)\n";
        return 1;
    }

    print_usage();

    // ---- Interactive loop ----
    std::string line;
    std::cout << "> " << std::flush;

    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            std::cout << "> " << std::flush;
            continue;
        }

        if (line == "/quit" || line == "/exit" || line == "/q") {
            break;
        }

        if (line == "/help") {
            print_usage();
            std::cout << "> " << std::flush;
            continue;
        }

        if (line == "/doc") {
            std::cout << "Document loaded. (stats available at startup)\n";
            std::cout << "> " << std::flush;
            continue;
        }

        std::cout << "\n[检索中...]\n" << std::flush;

        std::string answer = pipeline.query(line);

        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << answer << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        std::cout << "> " << std::flush;
    }

    std::cout << "\nBye.\n";
    return 0;
}
