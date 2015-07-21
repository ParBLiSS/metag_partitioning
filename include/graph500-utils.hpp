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
#include <memory> // shared ptr

#include "ccl.hpp"





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
    localVector.reserve( 2 * nedges + (1UL << scale));

    T src, dest;
    for (int i = 0; i < nedges; ++i) {
      src = edges[2*i];
      dest = edges[2*i+1];
      if ((src >= 0) && (dest >= 0)) {  // valid edge
        localVector.emplace_back(src, dest, src);
      } // -1 need to be ignored.
    }

    free(edges);
  }

};


class Graph500Converter {
public:

  template <typename T>
  static void generate(T const* edges, size_t nedges, size_t nverts,
			std::vector< ::std::tuple<T, T, T> > &localVector,
			MPI_Comm comm) {

    // nedges and result are both local
    int p;
    MPI_Comm_size(comm, &p);

    // copy into the local vector
    localVector.reserve(2 * nedges + nverts);
    T src, dest;
    for (int i = 0; i < nedges; ++i) {
      src = edges[2*i];
      dest = edges[2*i + 1];
      if ((src >= 0) && (dest >= 0)) {  // valid edge
        localVector.emplace_back(src, dest, src);
      } // -1 need to be ignored.
    }

  }
  
  template <typename Packed, typename T>
  static void generate(Packed const* edges, size_t nedges, size_t nverts,
				std::vector< ::std::tuple<T, T, T> > &localVector,
				MPI_Comm comm) {

  // nedges and result are both local
    int p;
    MPI_Comm_size(comm, &p);

    // copy into the local vector
    localVector.reserve( 2 * nedges + nverts);
    T src, dest;
    Packed const *curr = edges;
    for (int i = 0; i < nedges; ++i, ++curr) {
      src = get_v0_from_edge(curr);
      dest = get_v1_from_edge(curr);
      if ((src >= 0) && (dest >= 0)) {  // valid edge
        localVector.emplace_back(src, dest, src);
      } // -1 need to be ignored.
    }
  }

  /// convert from local storage (of type SpMat) given comm.
  template <typename DER, typename CommGrid, typename T>
  static void generate(DER & edges, size_t total_nverts, ::std::shared_ptr<CommGrid> commgrid,
        std::vector< ::std::tuple<T, T, T> > &localVector) {

    // get the comm.
    MPI_Comm comm = commgrid->GetWorld();
    int col_rank = commgrid->GetRankInProcCol();
    int row_rank = commgrid->GetRankInProcRow();

    // get the number of local edges.
    size_t nedges = edges.getnnz();  // num edges is number of non-zero entries.
//    size_t nverts = edges.getnzc();  // num non-zero columns
//    mxx::datatype<size_t> dt;
//    MPI_Allreduce(&nverts, &total_nverts, 1, dt.type(), MPI_SUM, comm);

    size_t vert_per_row = total_nverts / commgrid->GetGridRows();
    size_t vert_per_col = total_nverts / commgrid->GetGridCols();

    localVector.reserve( 2 * nedges + std::max(vert_per_row, vert_per_col));


    // data is distributed into a grid, 2D partition of the adjacency matrix.

    printf("rank %d comm grid col %d, row %d.  local nedges = %lu, global verts %lu.  vert_per_row %lu, col %lu\n",
           commgrid->GetRank(), col_rank, row_rank, nedges, total_nverts, vert_per_row, vert_per_col);

    T src, dest;
    // iterate over the the content of the local sparse matrix and to generate the distributed vector
    for (auto coliter = edges.begcol(), colend = edges.endcol(); coliter != colend; ++coliter) {
      dest = static_cast<T>(coliter.colid());  // local coord
      // change to global coordinates
      dest += (col_rank * vert_per_col);

      for (auto eliter = edges.begnz(coliter), elend = edges.endnz(coliter); eliter != elend; ++eliter) {
        src = static_cast<T>(eliter.rowid());  // local coord

        // change to global coordinates
        src += (row_rank * vert_per_row);

        if ((src >= 0) && (dest >= 0)) {  // valid edge
          localVector.emplace_back(src, dest, src);
        } // -1 need to be ignored.

      }
    }

//
//    for (int i= 0; i < localVector.size(); ++i) {
//      printf("rank %d: %ld %ld %ld \n", commgrid->GetRank(), std::get<0>(localVector[i]), std::get<1>(localVector[i]), std::get<2>(localVector[i]));
//    }

  }


};

///  the log sort based connected components algorithm requires self loop (to represent unconnected vertex, and to differentiate vertices with indegree or outdegree of 1 from the unconnected vertex)
///  it also needs a reverse edge for every forward edge.  note that we already allocated the vector's reserved space.
template <typename T>
void ensure_undirected_and_self_looping(std::vector<T> & vector) {
  // first find the unique vertices

  std::vector<T> verts = vector;
  std::sort(verts.begin(), verts.end(), [](T const &x, T const &y) {
    return std::get<kmerTuple::kmer>(x) < std::get<kmerTuple::kmer>(y);
  });

  auto end = std::unique(verts.begin(), verts.end(), [](T const &x, T const &y) {
    return std::get<kmerTuple::kmer>(x) == std::get<kmerTuple::kmer>(y);
  });
  verts.erase(end, verts.end());

  // now create the selfloops
  std::for_each(verts.begin(), verts.end(), [](T &x) {
    std::get<kmerTuple::Pc>(x) = std::get<kmerTuple::kmer>(x);
    std::get<kmerTuple::Pn>(x) = std::get<kmerTuple::kmer>(x);
  });


  // now create the inverse edges
  std::vector<T> reverse = vector;
  std::for_each(reverse.begin(), reverse.end(), [](T &x) {
    std::get<kmerTuple::kmer>(x) = std::get<kmerTuple::Pn>(x);
    std::get<kmerTuple::Pn>(x) = std::get<kmerTuple::Pc>(x);
    std::get<kmerTuple::Pc>(x) = std::get<kmerTuple::kmer>(x);

  });

  // now merge with the original vector.
  vector.insert(vector.end(), reverse.begin(), reverse.end());
  vector.insert(vector.end(), verts.begin(), verts.end());


}


#endif
