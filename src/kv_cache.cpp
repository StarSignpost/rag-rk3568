#include "rag/kv_cache.h"

#include <algorithm>
#include <stdexcept>

namespace rag {

PageManager::PageManager() {
    blocks_.reserve(KV_MAX_PAGES);
}

int PageManager::alloc_block(const std::vector<int>& tokens) {
    if ((int)blocks_.size() >= KV_MAX_PAGES) {
        // Try to evict unpinned, unreferenced blocks
        int evicted = evict_context_blocks(1);
        if (evicted == 0) {
            // Last resort: evict any unpinned block
            for (auto& b : blocks_) {
                if (!b.pinned && b.ref_count == 0) {
                    // Remove from hash map
                    for (auto it = hash_to_block_.begin(); it != hash_to_block_.end(); ++it) {
                        if (it->second.block_id == b.block_id) {
                            hash_to_block_.erase(it);
                            break;
                        }
                    }
                    b.kv_dma_buf = nullptr;
                    b.dirty = true;
                    return b.block_id;
                }
            }
            return -1;  // no free blocks
        }
    }

    int id = (int)blocks_.size();
    KVBlock b;
    b.block_id     = id;
    b.content_hash = page_hash64(tokens.data(), tokens.size());
    b.dirty        = true;
    b.ref_count    = 1;
    blocks_.push_back(b);
    return id;
}

std::vector<int> PageManager::map_tokens_to_blocks(const std::vector<int>& tokens) {
    std::vector<int> result;

    int num_pages = ((int)tokens.size() + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;

    for (int p = 0; p < num_pages; ++p) {
        int start = p * KV_PAGE_SIZE;
        int end   = std::min(start + KV_PAGE_SIZE, (int)tokens.size());
        std::vector<int> slice(tokens.begin() + start, tokens.begin() + end);

        uint64_t hash = page_hash64(slice.data(), slice.size());

        auto it = hash_to_block_.find(hash);
        if (it != hash_to_block_.end() && it->second.matches(slice)) {
            // Hash hit + exact match → reuse
            int bid = it->second.block_id;
            blocks_[bid].ref_count++;
            blocks_[bid].dirty = false;
            result.push_back(bid);
        } else {
            // Miss or collision → allocate new block
            int bid = alloc_block(slice);
            if (bid < 0) {
                // Allocation failed, mark as dirty (will be prefilled)
                result.push_back(-1);
                continue;
            }
            hash_to_block_[hash] = KVCacheTag{bid, slice};
            result.push_back(bid);
        }
    }

    return result;
}

std::vector<int> PageManager::collect_prefill_tokens(
    const std::vector<int>& block_ids,
    const std::vector<int>& all_tokens) const {

    std::vector<int> prefill_tokens;

    for (int i = 0; i < (int)block_ids.size(); ++i) {
        int bid = block_ids[i];
        if (bid < 0 || bid >= (int)blocks_.size()) continue;
        if (!blocks_[bid].dirty) continue;

        int start = i * KV_PAGE_SIZE;
        int end   = std::min(start + KV_PAGE_SIZE, (int)all_tokens.size());
        prefill_tokens.insert(prefill_tokens.end(),
                              all_tokens.begin() + start,
                              all_tokens.begin() + end);
    }

    return prefill_tokens;
}

void PageManager::pin_block(int block_id) {
    if (block_id >= 0 && block_id < (int)blocks_.size()) {
        blocks_[block_id].pinned = true;
    }
}

void PageManager::release_block(int block_id) {
    if (block_id < 0 || block_id >= (int)blocks_.size()) return;
    auto& b = blocks_[block_id];
    if (b.ref_count > 0) b.ref_count--;
}

int PageManager::evict_context_blocks(int count) {
    int evicted = 0;
    for (auto& b : blocks_) {
        if (evicted >= count) break;
        if (!b.pinned && b.ref_count == 0) {
            // Remove from hash map
            for (auto it = hash_to_block_.begin(); it != hash_to_block_.end(); ++it) {
                if (it->second.block_id == b.block_id) {
                    hash_to_block_.erase(it);
                    break;
                }
            }
            b.kv_dma_buf = nullptr;
            b.dirty = true;
            evicted++;
        }
    }
    return evicted;
}

int PageManager::num_free_slots() const {
    return KV_MAX_PAGES - (int)blocks_.size();
}

bool PageManager::is_block_pinned(int block_id) const {
    if (block_id < 0 || block_id >= (int)blocks_.size()) return false;
    return blocks_[block_id].pinned;
}

} // namespace rag
