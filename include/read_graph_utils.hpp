/**
 * @file    read_graph_utils.hpp
 * @ingroup
 * @author  tpan
 * @brief
 * @details
 *
 * Copyright (c) 2015 Georgia Institute of Technology.  All Rights Reserved.
 *
 * TODO add License
 */
#ifndef INCLUDE_READ_GRAPH_UTILS_HPP_
#define INCLUDE_READ_GRAPH_UTILS_HPP_


#include <vector>
#include "configParam.hpp"
#include <mpi.h>
#include <utility>  // declval
#include <type_traits>  // remove_reference
#include <limits>   // numeric_limits
#include "mxx/sort.hpp"
#include "mxx/timer.hpp"
#include "sortTuples.hpp"
#include <tuple>    // std::get

class ReadGraphGenerator
{
public:
  template <typename tuple_t>
  static void generate(::std::vector<tuple_t> &localVector, MPI_Comm comm) {

    static constexpr int Pn = kmerTuple::Pn;
    static constexpr int Pc = kmerTuple::Pc;

    using PID_TYPE = typename std::remove_reference<decltype(std::get<kmerTuple::Pc>(std::declval<tuple_t>()))>::type;

    static constexpr PID_TYPE PID_MAX = std::numeric_limits<PID_TYPE>::max();

    // get communicaiton parameters
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    /*
     * Indices inside tuple will go like this:
     * 0 : KmerId
     * 1 : P_new
     * 2 : P_old
     */

    MP_TIMER_START();

    assert(localVector.size() > 0);
    auto start = localVector.begin();
    auto end = localVector.end();


    //Sort tuples by KmerId
    // sort by k-mer and use min Pc for a kmer to create a star-graph between the reads based on shared kmers.

    // sort by k-mers and update Pn
    mxx::sort(start, end, layer_comparator<kmerTuple::kmer, tuple_t>(), comm, false);
    typedef KmerReduceAndMarkAsInactive<tuple_t> KmerReducerType;
    KmerReducerType r1;
    r1(start, end, comm);
    MP_TIMER_END_SECTION("iteration KMER phase completed");


    // after k-mer reduce, replace MAX-1 or MAX with pc:
    // TODO: own k-mer reduction function!
    for (auto it = localVector.begin(); it != localVector.end(); ++it)
    {
      if (std::get<Pn>(*it) >= (PID_MAX - 1))
        std::get<Pn>(*it) = std::get<Pc>(*it);
    }
  }
};

#endif /* INCLUDE_READ_GRAPH_UTILS_HPP_ */
