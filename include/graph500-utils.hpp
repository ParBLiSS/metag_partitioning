/**
 * @file    graph500-utils.hpp
 * @ingroup group
 * @author  Tony Pan<tpan7@gatech.edu>
 * @brief   Implements the de bruijn graph clustering in log(D_max) time.
 *          This file is only used for benchmarking partitioning algorithm in SC15. 
 *          For application use, please see src/getHistogram.cpp
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */
#ifndef METAG_GRAPH500_UTILS
#define METAG_GRAPH500_UTILS


//Includes
#include <mpi.h>
#include <iostream>

//Own includes
#include "sortTuples.hpp"
#include "configParam.hpp"

#include <mxx/collective.hpp>
#include <mxx/distribution.hpp>
#include <mxx/reduction.hpp>
#include <mxx/timer.hpp>

#include <sstream>

using namespace std;

#include <generator/make_graph.h>
#include <tuple>
#include <vector>

class Graph500Generator {
public:
  template <typename T>
  static void generate(cmdLineParamsGraph500 &cmdLineVals, 
			std::vector< ::std::tuple<T, T, T> > &localVector,
			MPI_Comm comm) {

    size_t scale = cmdLineVals.scale;
    size_t edgefactor = cmdLineVals.edgefactor;

    // nedges and result are both local
    int64_t nedges;
    T* edges;

    double initiator[] = {.57, .19, .19, .05};

    make_graph(scale, edgefactor * (1UL << scale), 1, 2, initiator, &nedges, &edges );

    int p;
    MPI_Comm_size(comm, &p);

    // copy into the local vector
    localVector.reserve(nedges);
    T id;
    for (int i = 0; i < nedges; ++i) {
      id = edges[2*i];
      if (id != -1) {  // valid edge
        localVector.emplace_back(id, id, edges[2*i+1]);
      } // -1 need to be ignored.
    }

    free(edges);
  }

};


class Graph500Converter {
public:

  template <typename T>
  static void generate(T const* edges, size_t nedges, 
			std::vector< ::std::tuple<T, T, T> > &localVector,
			MPI_Comm comm) {

    // nedges and result are both local
    int p;
    MPI_Comm_size(comm, &p);

    // copy into the local vector
    localVector.reserve(nedges);
    T id;
    for (int i = 0; i < nedges; ++i) {
      id = edges[2*i];
      if (id != -1) {  // valid edge
        localVector.emplace_back(id, id, edges[2*i+1]);
      } // -1 need to be ignored.
    }

  }
  
  template <typename Packed, typename T>
  static void generate_from_packed(Packed const* edges, size_t nedges,
				std::vector< ::std::tuple<T, T, T> > &localVector,
				MPI_Comm comm) {

  // nedges and result are both local
    int p;
    MPI_Comm_size(comm, &p);

    // copy into the local vector
    localVector.reserve(nedges);
    T id;
    Packed const *curr = edges;
    for (int i = 0; i < nedges; ++i, ++curr) {
      id = get_v0_from_edge(curr);
      if (id != -1) {  // valid edge
        localVector.emplace_back(id, id, get_v1_from_edge(curr));
      } // -1 need to be ignored.
    }



  }

};



#endif
