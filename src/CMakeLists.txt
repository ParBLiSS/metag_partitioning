cmake_minimum_required(VERSION 2.6)

# project settings
project(Metagenomics-src)

add_executable(statAndCompare statAndCompare.cpp)
target_link_libraries(statAndCompare ${EXTRA_LIBS})

add_executable(getHistogram getHistogram.cpp)
target_link_libraries(getHistogram bliss-common)
target_link_libraries(getHistogram ${EXTRA_LIBS})

add_executable(log-sort logSort.cpp)
target_link_libraries(log-sort bliss-common)
target_link_libraries(log-sort ${EXTRA_LIBS})

add_executable(log-sort-graph500 logSortGraph500.cpp)
target_link_libraries(log-sort-graph500 ${CMAKE_SOURCE_DIR}/ext/graph500/generator/libgraph_generator_mpi.a)
target_link_libraries(log-sort-graph500 ${EXTRA_LIBS})
