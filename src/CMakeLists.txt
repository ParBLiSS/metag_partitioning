cmake_minimum_required(VERSION 2.6)

# project settings
project(Metagenomics-src)

add_executable(statAndCompare statAndCompare.cpp)
target_link_libraries(statAndCompare ${EXTRA_LIBS})

add_executable(metaG assembly.cpp)
target_link_libraries(metaG ${EXTRA_LIBS})

add_executable(log-sort logSort.cpp)
target_link_libraries(log-sort ${EXTRA_LIBS})

add_executable(log-sort-graph500 logSortGraph500.cpp)
target_link_libraries(log-sort-graph500 ${CMAKE_BINARY_DIR}/lib/libGraphGenlib.a)
target_link_libraries(log-sort-graph500 ${EXTRA_LIBS})
