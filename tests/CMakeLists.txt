add_executable(test_bm25 test_bm25.cpp)
target_link_libraries(test_bm25 rag)
add_test(NAME bm25 COMMAND test_bm25)

add_executable(test_retriever test_retriever.cpp)
target_link_libraries(test_retriever rag)
add_test(NAME retriever COMMAND test_retriever)

add_executable(test_kv_cache test_kv_cache.cpp)
target_link_libraries(test_kv_cache rag)
add_test(NAME kv_cache COMMAND test_kv_cache)
