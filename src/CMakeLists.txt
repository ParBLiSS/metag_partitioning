cmake_minimum_required(VERSION 2.6)

# project settings
project(Metagenomics-src)

add_executable(parallel_fastq_iterate run_fastq_iterate.cpp)
target_link_libraries(parallel_fastq_iterate ${EXTRA_LIBS})
target_link_libraries(parallel_fastq_iterate bliss-common)

add_executable(build_Kmer_tuples build_Kmer_tuples.cpp)
target_link_libraries(build_Kmer_tuples ${EXTRA_LIBS})
target_link_libraries(build_Kmer_tuples bliss-common)


