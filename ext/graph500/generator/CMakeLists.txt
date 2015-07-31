# Top level directory has the include files

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized --std=c++11 -DGRAPH_GENERATOR_MPI -DGRAPHGEN_DISTRIBUTED_MEMORY -ffast-math -O3 -Drestrict=__restrict__")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wuninitialized  -DGRAPH_GENERATOR_MPI -DGRAPHGEN_DISTRIBUTED_MEMORY -ffast-math -O3 -Drestrict=__restrict__")

ADD_LIBRARY( GraphGenlib STATIC btrd_binomial_distribution.c splittable_mrg.c mrg_transitions.c graph_generator.c permutation_gen.c make_graph.c utils.c scramble_edges.c  apply_permutation_mpi.c )
