cmake_minimum_required(VERSION 2.6)

# project settings
project(Metagenomics-src)

# timer shared lib
add_library(mp-timer timer.cpp)
target_link_libraries(mp-timer rt)

add_executable(parallel_fastq_iterate run_fastq_iterate.cpp)
target_link_libraries(parallel_fastq_iterate ${EXTRA_LIBS})
target_link_libraries(parallel_fastq_iterate bliss-common)

add_executable(sort_merge_tuples sort_merge_tuples.cpp)
target_link_libraries(sort_merge_tuples ${EXTRA_LIBS})
target_link_libraries(sort_merge_tuples bliss-common)
target_link_libraries(sort_merge_tuples mp-timer)


