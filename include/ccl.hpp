/**
 * @file    ccl.hpp
 * @ingroup group
 * @author  Tony Pan<tpan7@gatech.edu>
 * @brief   Implements the de bruijn graph clustering in log(D_max) time.
 *          This file is only used for benchmarking partitioning algorithm in SC15. 
 *          For application use, please see src/getHistogram.cpp
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */
#ifndef METAG_CCL
#define METAG_CCL

//Includes
#include <mpi.h>

//Own includes
#include "sortTuples.hpp"
#include "configParam.hpp"

#include <mxx/collective.hpp>
#include <mxx/distribution.hpp>
#include <mxx/reduction.hpp>
#include <mxx/timer.hpp>

#include <sstream>

using namespace std;

#include <tuple>
#include <vector>
#include <type_traits>  // remove_reference
#include <utility>      // declval
#include <limits>       // numeric_limits


template <typename tuple_t>
void dump_seeds(std::vector<tuple_t> const &seeds, MPI_Comm comm, std::string const &seedFile) {
	
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    // gather the seeds
    std::vector<tuple_t> allseeds;
    if (p > 1)
      mxx::gather_vectors(seeds, comm).swap(allseeds);


    // rank 0 write out the seeds
    if (rank == 0) {
      std::ofstream ofs(seedFile.c_str());
      std::for_each(allseeds.begin(), allseeds.end(), [&ofs](tuple_t const& x) { ofs << std::get<kmerTuple::Pc>(x) << std::endl; });

      ofs.close();
      printf("partition count = %lu. seeds written to %s\n", allseeds.size(), seedFile.c_str());
    }
}
template <typename tuple_t>
void dump_vector(std::vector<tuple_t> const &seeds, MPI_Comm comm, std::string const &seedFile) {
	
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    std::stringstream ss;
    ss << seedFile << "." << rank;

      std::ofstream ofs(ss.str().c_str());
      std::for_each(seeds.begin(), seeds.end(), [&ofs](tuple_t const& x) { 
        ofs << std::get<kmerTuple::kmer>(x) << " " <<
              std::get<kmerTuple::Pn>(x) << " " <<
              std::get<kmerTuple::Pc>(x) << std::endl; });

      ofs.close();
      printf("rank %d partition count = %lu. seeds written to %s\n", rank, seeds.size(), ss.str().c_str());
}

template <typename T, typename Comparator, typename Equal>
void unique(std::vector<T> &vector, Comparator const &less, Equal const &eq, MPI_Comm comm) {
  // use last seed as splitter
  int rank;
  int p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);
//
//  if (rank == 0)
//    printf("UNIQUE_COPY\n");



//  for (int i= 0; i < vector.size(); ++i) {
//    printf("before sort rank %d: %ld %ld %ld \n", rank,
//       std::get<kmerTuple::kmer>(vector[i]),
//       std::get<kmerTuple::Pn>(vector[i]),
//       std::get<kmerTuple::Pc>(vector[i]));
//  }

  assert(vector.size() > 0);

  // sort
  if (p > 1) {
    mxx::block_decompose(vector, comm);

    // mxx_sort
    mxx::sort(vector.begin(), vector.end(), less, comm, false);

  } else {
    std::sort(vector.begin(), vector.end(), less);
  }
//  //printf("rank %d vert count : %lu first %ld last %ld\n", rank, vector.size(), std::get<kmerTuple::Pc>(vector.front()), std::get<kmerTuple::Pc>(vector.back()));
//    for (int i= 0; i < vector.size(); ++i) {
//      printf("after sort rank %d: %ld %ld %ld \n", rank,
//         std::get<kmerTuple::kmer>(vector[i]),
//         std::get<kmerTuple::Pn>(vector[i]),
//         std::get<kmerTuple::Pc>(vector[i]));
//    }

  // local unique.
  auto end = std::unique(vector.begin(), vector.end(), eq);
  vector.erase(end, vector.end());


//  //printf("rank %d vert count : %lu seeds count: %lu\n", rank, vector.size(), seeds.size());
//  for (int i= 0; i < seeds.size(); ++i) {
//    printf("after local unique rank %d: %ld %ld %ld \n", rank, std::get<kmerTuple::kmer>(seeds[i]),
//           std::get<kmerTuple::Pn>(seeds[i]), std::get<kmerTuple::Pc>(seeds[i]));
//  }

  // check if cross boundaries boundaries.
  if (p > 1) {

    std::vector<T> splitters;
    if ((vector.size() > 0) && (rank > 0)) {
      splitters.emplace_back(vector.front());
    }
    mxx::allgatherv(splitters, comm).swap(splitters);

    // splitters should contain unique values only
    auto uend = std::unique(splitters.begin(), splitters.end(), eq);
    splitters.erase(uend, splitters.end());

//    //printf("rank %d vert count : %lu vector count: %lu\n", rank, vector.size(), vector.size());
//    for (int i= 0; i < splitters.size(); ++i) {
//      printf("spliters rank %d: %ld %ld %ld \n", rank, std::get<kmerTuple::kmer>(splitters[i]),
//             std::get<kmerTuple::Pn>(splitters[i]), std::get<kmerTuple::Pc>(splitters[i]));
//    }

    //== compute the send count
    std::vector<int> send_counts(p, 0);
    if (vector.size() > 0) {
      //== since both sorted, search map entry in buffer - mlog(n).  the otherway around is nlog(m).
      // note that map contains splitters.  range defined as [map[i], map[i+1]), so when searching in buffer using entry in map, search via lower_bound
      auto b = vector.begin();
      auto e = b;
      for (int i = 0; i < splitters.size(); ++i) {
        e = ::std::lower_bound(b, vector.end(), splitters[i], less);
        send_counts[i] = ::std::distance(b, e);
        b = e;
      }
      // last 1
      send_counts[splitters.size()] = ::std::distance(b, vector.end());
    }

//    for (int i = 0; i < p; ++i) {
//      printf("rank %d send to %d %d\n", rank, i, send_counts[i]);
//    }

    // a2a to redistribute vector so we can do unique one more time.
    mxx::all2all(vector, send_counts, comm).swap(vector);


//    //printf("rank %d vert count : %lu vector count: %lu\n", rank, vector.size(), vector.size());
//    for (int i= 0; i < vector.size(); ++i) {
//      printf("after all2all rank %d: %ld %ld %ld \n", rank, std::get<kmerTuple::kmer>(vector[i]),
//             std::get<kmerTuple::Pn>(vector[i]), std::get<kmerTuple::Pc>(vector[i]));
//    }


    // local unique.  (still sorted)
    auto newend = std::unique(vector.begin(), vector.end(), eq);
    vector.erase(newend, vector.end());

//    printf("rank %d vert count : %lu seeds final count: %lu\n", rank, vector.size(), seeds.size());

  }

//  //printf("rank %d vert count : %lu seeds count: %lu\n", rank, vector.size(), seeds.size());
//  for (int i= 0; i < seeds.size(); ++i) {
//    printf("after global unique rank %d: %ld %ld %ld \n", rank, std::get<kmerTuple::kmer>(seeds[i]),
//           std::get<kmerTuple::Pn>(seeds[i]), std::get<kmerTuple::Pc>(seeds[i]));
//  }

}


template <typename T, typename Comparator, typename Equal>
std::vector<T> unique_copy(std::vector<T> const &vector, Comparator const &less, Equal const &eq, MPI_Comm comm) {

  std::vector<T> uniq = vector;
  unique(uniq, less, eq, comm);
  return uniq;
}



template <typename T>
std::vector<T> get_partition_seeds(std::vector<T> &vector, MPI_Comm comm) {
//  for (int i= 0; i < vector.size(); ++i) {
//    printf("before SEEDS rank %d: %ld %ld %ld \n", rank, std::get<kmerTuple::kmer>(vector[i]), 
//       std::get<kmerTuple::Pn>(vector[i]), std::get<kmerTuple::Pc>(vector[i]));
//  }
//  int rank;
//  MPI_Comm_rank(comm, &rank);
//  if (rank == 0)
//    printf("seeds\n");



  //Lets ensure Pn and Pc are equal for every tuple
  //This was not ensured during the program run
  std::for_each(vector.begin(), vector.end(), [](T &t){ std::get<kmerTuple::Pn>(t) = std::get<kmerTuple::Pc>(t);});

  // sort and find unique in distributed environment.
  return unique_copy(vector, [](const T& x, const T&y){
    return std::get<kmerTuple::Pc>(x) < std::get<kmerTuple::Pc>(y);
  }, [](const T& x, const T&y){
    return std::get<kmerTuple::Pc>(x) == std::get<kmerTuple::Pc>(y);
  }, comm);

}




template <typename T>
std::vector<T> get_partition_seeds_to_all(std::vector<T> &vector, MPI_Comm comm) {

  // use last seed as splitter
  int rank;
  int p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);
//
//  for (int i= 0; i < vector.size(); ++i) {
//    printf("before SEEDS rank %d: %ld %ld %ld \n", rank, 
//       std::get<kmerTuple::kmer>(vector[i]), 
//       std::get<kmerTuple::Pn>(vector[i]),
//       std::get<kmerTuple::Pc>(vector[i]));
//  }
//

//  if (rank == 0)
//    printf("seeds to all\n");


  auto temp = get_partition_seeds(vector, comm);

  // get the sizes
  size_t count = temp.size();
  std::vector<size_t> recv_counts(p, 0);
  mxx::datatype<size_t> count_dt;
  MPI_Allgather(&count, 1, count_dt.type(), &(recv_counts[0]), 1, count_dt.type(), comm);


  // compute the displacements
  std::vector<size_t> recv_displs = mxx::get_displacements(recv_counts);

  // allocate
  std::vector<T> seeds(recv_displs[p-1] + recv_counts[p-1]);

  // allgatherv
  mxx::datatype<T> dt;
  MPI_Allgatherv(&(temp[0]), count, dt.type(), &(seeds[0]), recv_counts, recv_displs, dt.type(), comm);

  return seeds;

}



template <typename T>
void unique(std::vector<T> &vector, MPI_Comm comm) {

//  int rank;
//  MPI_Comm_rank(comm, &rank);
//  if (rank == 0)
//    printf("UNIQUE\n");

  unique(vector, [](const T& x, const T&y){
    return (std::get<kmerTuple::Pc>(x) < std::get<kmerTuple::Pc>(y)) ? true :
      ( (std::get<kmerTuple::Pc>(x) == std::get<kmerTuple::Pc>(y)) && (std::get<kmerTuple::Pn>(x) < std::get<kmerTuple::Pn>(y)) ? true :
      ( (std::get<kmerTuple::Pc>(x) == std::get<kmerTuple::Pc>(y)) && (std::get<kmerTuple::Pn>(x) == std::get<kmerTuple::Pn>(y)) && (std::get<kmerTuple::kmer>(x) < std::get<kmerTuple::kmer>(y)) ? true : false ) ) ;
  }, [](const T& x, const T&y){
    return (std::get<kmerTuple::Pc>(x) == std::get<kmerTuple::Pc>(y)) &&
      (std::get<kmerTuple::Pn>(x) == std::get<kmerTuple::Pn>(y)) &&
      (std::get<kmerTuple::kmer>(x) == std::get<kmerTuple::kmer>(y));
  }, comm);
}


/// parallel/MPI log(D_max) implementation
/// localVector should be a list of undirected edges (reverse of each edge is present as well)
template <typename tuple_t>
void cluster_reads_par(::std::vector<tuple_t> &localVector, MPI_Comm comm)
{
  static constexpr int Pn = kmerTuple::Pn;
  static constexpr int Pc = kmerTuple::Pc;

  using PID_TYPE = typename std::remove_reference<decltype(std::get<kmerTuple::Pc>(std::declval<tuple_t>()))>::type;

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n";
  }

  mxx::timer t;
  double startTime = t.elapsed();

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  MP_TIMER_START();

  // keep only unique entries from the local vector.
  unique(localVector, comm);

  assert(localVector.size() > 0);

  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;

  layer_comparator<Pc, tuple_t> pc_comp;

  while (keepGoing) {
    mxx::sort(localVector.begin(), localVector.end(),
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<Pc>(x) < std::get<Pc>(y)
            || (std::get<Pc>(x) == std::get<Pc>(y)
            && std::get<Pn>(x) < std::get<Pn>(y)));
        }, comm, false);
    MP_TIMER_END_SECTION("mxx::sort");


    // scan and find new p
    auto begin = localVector.begin();
    auto end = localVector.end();
    std::vector<tuple_t> newtuples;
    bool done = true;

    std::size_t local_size = end - begin;
    std::vector<std::size_t> distr = mxx::allgather(local_size, comm);
    if (rank == 0) {
      //std::cout << "local_sizes: [MAX : " << *std::max_element(distr.begin(), distr.end()) << ", MIN: " << *std::min_element(distr.begin(), distr.end()) << ", SUM: " << std::accumulate(distr.begin(), distr.end(), 0) << "]" << std::endl;
      //std::cout << "local_sizes: " << distr << std::endl;
    }

    // find last bucket start and send across boundaries!
    tuple_t last_min = *(end-1);
    last_min = *std::lower_bound(begin, end, last_min, pc_comp);
    // for each processor, get the first element of the last most element
    tuple_t prev_min = mxx::exscan(last_min,
        [](const tuple_t& x, const tuple_t& y){
          // return max Pc, if equal, return min Pn.  sorted by Pc + scan with max Pc == no op.  this finds min Pn when equal Pc.
          if (std::get<Pc>(x) < std::get<Pc>(y) ||
            (std::get<Pc>(x) == std::get<Pc>(y)
             && std::get<Pn>(x) > std::get<Pn>(y)))
          return y;
          else return x;}
        , comm);
    tuple_t prev_el = mxx::right_shift(*(end-1), comm);

    // get the next element
    tuple_t first_max = *(std::upper_bound(begin, end, *begin, pc_comp)-1);
    tuple_t next_max = mxx::reverse_exscan(first_max,
        [](const tuple_t& x, const tuple_t& y){
          // return min Pc, if equal, return max Pn
          if (std::get<Pc>(x) > std::get<Pc>(y) ||
            (std::get<Pc>(x) == std::get<Pc>(y)
             && std::get<Pn>(x) < std::get<Pn>(y)))
          return y;
          else return x;}
        , comm);
    //tuple_t next_el = mxx::left_shift(*begin, comm);

    MP_TIMER_END_SECTION("reductions");

    // for each range/bucket
    for(; begin != end; ) {
      // find bucket (linearly)
      auto eqr = findRange(begin, end, *begin, pc_comp);
      assert(eqr.first == begin);

      // get smallest Pn in bucket
      auto min_pn = std::get<Pn>(*eqr.first);
      if (rank > 0 && std::get<Pc>(prev_min) == std::get<Pc>(*eqr.first)) {
        // first bucket and it starts on the processor to the left
        min_pn = std::get<Pn>(prev_min);
      }

      // get largest Pn in bucket
      auto max_pn = std::get<Pn>(*(eqr.second-1));
      if (rank < p-1 && std::get<Pc>(next_max) == std::get<Pc>(*eqr.first)) {
          max_pn = std::get<Pn>(next_max);
      }

      // remove single element buckets
      // TCP: if tuple range crosses process boundary with 1 element in first process, it will have the min Pn already.
      // TCP: problem:  if we have 2 vertices referring to each other only, then this will set each as separate.
      //    FIX: only set Pc to Pn if Pn is lower.  if not lower, then we have a self loop, or a real loop between 2 or more vertes,
      //    and the updated Pc with smaller Pn will be sorted to be together with the pn.  to resolve this, add self loops to all nodes.
      if (eqr.first + 1 == eqr.second && (rank == 0
          || (rank > 0 && std::get<Pc>(*eqr.first) != std::get<Pc>(prev_el)))) {
        // single element -> no need to stick around
        std::get<Pc>(*eqr.first) = std::get<Pn>(*eqr.first);
        begin = eqr.second;
        continue;
      }



      // check if all elements of the bucket are identical -> set them to
      // their Pn.
      if (min_pn == max_pn) {
          for (auto it = eqr.first; it != eqr.second; ++it) {
            std::get<Pc>(*it) = std::get<Pn>(*it);
          }
          begin = eqr.second;
          continue;
      }

      done = false;

      // for each further element:
      bool found_flip = false;
      PID_TYPE prev_pn = std::get<Pn>(prev_el);
      auto it = eqr.first;
      if (rank == 0 || (rank > 0 && std::get<Pc>(*eqr.first) != std::get<Pc>(prev_el))) {
          // skip first (since it is the min entry
          prev_pn = min_pn;
          it++;
      }
      for (; it != eqr.second; ++it) {
        PID_TYPE next_pn = std::get<Pn>(*it);
        // if duplicate or both entries are equal (Pn==Pc)
        if (std::get<Pn>(*it) == prev_pn || std::get<Pn>(*it) == std::get<Pc>(*it)) {
          if (!found_flip) {
            // set flipped
            found_flip = true;
            std::get<Pn>(*it) = std::get<Pc>(*it);
            std::get<Pc>(*it) = min_pn;
          } else {
            // moves over to new partition (since duplicates are not needed)
            std::get<Pn>(*it) = min_pn;
            std::get<Pc>(*it) = min_pn;
          }
        } else {
          // if either Pn > Pc or Pn < Pc, we flip the entry and return it
          // to `Pn` with the new minimum
          // update tuple, set new min and flip
          std::swap(std::get<Pn>(*it),std::get<Pc>(*it));
          std::get<Pn>(*it) = min_pn;
        }
        prev_pn = next_pn;
      }

      if (!found_flip) {
        // TODO: don't do this for the first or last bucket...
        // TODO: we need only one flipped per bucket
        tuple_t t = *eqr.first;
        std::swap(std::get<Pn>(t),std::get<Pc>(t));
        newtuples.push_back(t);
      }

      // next range
      begin = eqr.second;
    }
    MP_TIMER_END_SECTION("local flips");
    localVector.insert(localVector.end(), newtuples.begin(), newtuples.end());
    MP_TIMER_END_SECTION("vector inserts");

    // check if all processors are done
    keepGoing = !mxx::test_all(done);
    MP_TIMER_END_SECTION("check termination");

    countIterations++;
    if(!rank)
      std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";
  }


  double time = t.elapsed() - startTime;


    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }
}

/// parallel/MPI log(D_max) implementation with removal of inactive partitions
/// localVector should be undirected (i.e. each edge has its reverse in there as well)
template <typename tuple_t>
void cluster_reads_par_inactive(bool load_balance, std::vector<tuple_t> &localVector, MPI_Comm comm)
{
  static constexpr int Pn = kmerTuple::Pn;
  static constexpr int Pc = kmerTuple::Pc;

  using PID_TYPE = typename std::remove_reference<decltype(std::get<kmerTuple::Pc>(std::declval<tuple_t>()))>::type;

  static constexpr PID_TYPE PID_MAX = std::numeric_limits<PID_TYPE>::max();

  //Specify Kmer Type
  static constexpr PID_TYPE inactive_partition = PID_MAX;

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);


  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n";
  }


  mxx::timer t;
  double startTime = t.elapsed();

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  MP_TIMER_START();


  assert(localVector.size() > 0);


  // keep only unique entries from the local vector.
  unique(localVector, comm);

  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;

  auto pend = localVector.end();

  while (keepGoing) {
    mxx::sort(localVector.begin(), pend,
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<Pc>(x) < std::get<Pc>(y)
            || (std::get<Pc>(x) == std::get<Pc>(y)
            && std::get<Pn>(x) < std::get<Pn>(y)));
        }, comm, false);
    MP_TIMER_END_SECTION("mxx::sort");

    layer_comparator<Pc, tuple_t> pc_comp;

    // scan and find new p
    auto begin = localVector.begin();
    //auto end = localVector.end();
    std::vector<tuple_t> newtuples;
    bool done = true;

    // TODO: if a local size becomes 0, then there might be an issue with the
    // min/max reductions:

    // TODO: something is still wrong!
    std::size_t local_size = pend - localVector.begin();
    std::vector<std::size_t> distr = mxx::allgather(local_size, comm);
    if (rank == 0) {
      //std::cout << "local_sizes: [MAX : " << *std::max_element(distr.begin(), distr.end()) << ", MIN: " << *std::min_element(distr.begin(), distr.end()) << ", SUM: " << std::accumulate(distr.begin(), distr.end(), 0) << "]" << std::endl;
      //std::cout << "local_sizes: " << distr << std::endl;
    }

    int color = local_size == 0 ? 0 : 1;
    MPI_Comm nonempty_comm;
    MPI_Comm_split(comm, color, rank, &nonempty_comm);
    int active_rank, active_p;
    MPI_Comm_rank(nonempty_comm, &active_rank);
    MPI_Comm_size(nonempty_comm, &active_p);

    tuple_t last_min, prev_min, prev_el, first_max, next_max, next_el;
    if (local_size != 0) {
      // find last bucket start and send across boundaries!
      last_min = *(pend-1);
      last_min = *std::lower_bound(begin, pend, last_min, pc_comp);
      // for each processor, get the first element of the last most element
      prev_min = mxx::exscan(last_min,
          [](const tuple_t& x, const tuple_t& y){
            // return max Pc, if equal, return min Pn
            if (std::get<Pc>(x) < std::get<Pc>(y) ||
              (std::get<Pc>(x) == std::get<Pc>(y)
               && std::get<Pn>(x) > std::get<Pn>(y)))
            return y;
            else return x;}
          , nonempty_comm);
      prev_el = mxx::right_shift(*(pend-1), nonempty_comm);

      // get the next element
      first_max = *(std::upper_bound(begin, pend, *begin, pc_comp)-1);
      next_max = mxx::reverse_exscan(first_max,
          [](const tuple_t& x, const tuple_t& y){
            // return min Pc, if equal, return max Pn
            if (std::get<Pc>(x) > std::get<Pc>(y) ||
              (std::get<Pc>(x) == std::get<Pc>(y)
               && std::get<Pn>(x) < std::get<Pn>(y)))
            return y;
            else return x;}
          , nonempty_comm);
      next_el = mxx::left_shift(*begin, nonempty_comm);
    }

    MPI_Comm_free(&nonempty_comm);

    MP_TIMER_END_SECTION("reductions");

    // for each range/bucket
    for(; begin != pend; ) {
      // find bucket (linearly)
      auto eqr = findRange(begin, pend, *begin, pc_comp);
      assert(eqr.first == begin);

      // get smallest Pn in bucket
      auto min_pn = std::get<Pn>(*eqr.first);
      if (active_rank > 0 && std::get<Pc>(prev_min) == std::get<Pc>(*eqr.first)) {
        // first bucket and it starts on the processor to the left
        min_pn = std::get<Pn>(prev_min);
      }

      // get largest Pn in bucket
      auto max_pn = std::get<Pn>(*(eqr.second-1));
      if (active_rank < active_p-1 && std::get<Pc>(next_max) == std::get<Pc>(*eqr.first)) {
          max_pn = std::get<Pn>(next_max);
      }

      // remove single element buckets
      if (eqr.first + 1 == eqr.second && (active_rank == 0
          || (active_rank > 0 && std::get<Pc>(*eqr.first) != std::get<Pc>(prev_el)))) {
        // single element -> no need to stick around
        if (std::get<Pn>(*eqr.first) == inactive_partition-1)
          //std::get<Pn>(*eqr.first) = std::get<Pc>(*eqr.first);
          std::get<Pn>(*eqr.first) = inactive_partition;
        else
          std::get<Pc>(*eqr.first) = std::get<Pn>(*eqr.first);
        begin = eqr.second;
        continue;
      }

      // check if all elements of the bucket are identical -> set them to
      // their Pn.
      if (min_pn == max_pn) {
          if (max_pn == inactive_partition-1)
          {
            // finished!
            for (auto it = eqr.first; it != eqr.second; ++it) {
              std::get<Pn>(*it) = inactive_partition;
            }
          }
          else if (std::get<Pc>(*eqr.first) == max_pn) {
            // finished, but need to participate in one more round
            // with one of my tuples
            for (auto it = eqr.first; it != eqr.second; ++it) {
              std::get<Pn>(*it) = inactive_partition-1;
            }
          } else {
            for (auto it = eqr.first; it != eqr.second; ++it) {
              std::get<Pc>(*it) = std::get<Pn>(*it);
            }
          }
          begin = eqr.second;
          continue;
      }

      if (min_pn > std::get<Pc>(*eqr.first))
        min_pn = std::get<Pc>(*eqr.first);

      done = false;

      // for each further element:
      bool found_flip = false;
      PID_TYPE prev_pn = std::get<Pn>(prev_el);
      auto it = eqr.first;
      if (rank == 0 || (rank > 0 && std::get<Pc>(*eqr.first) != std::get<Pc>(prev_el))) {
          if (std::get<Pn>(*eqr.first) > min_pn)
            std::get<Pn>(*eqr.first) = min_pn;
          // skip first (since it is the min entry
          prev_pn = min_pn;
          it++;
      }
      for (; it != eqr.second; ++it) {
        if (std::get<Pn>(*it) == inactive_partition-1)
          std::get<Pn>(*it) = std::get<Pc>(*it);
        PID_TYPE next_pn = std::get<Pn>(*it);
        // if duplicate or both entries are equal (Pn==Pc)
        if (std::get<Pn>(*it) == prev_pn || std::get<Pn>(*it) == std::get<Pc>(*it)) {
          if (!found_flip) {
            // set flipped
            found_flip = true;
            std::get<Pn>(*it) = std::get<Pc>(*it);
            std::get<Pc>(*it) = min_pn;
          } else {
            // moves over to new partition (since duplicates are not needed)
            std::get<Pn>(*it) = min_pn;
            std::get<Pc>(*it) = min_pn;
          }
        } else {
          // if either Pn > Pc or Pn < Pc, we flip the entry and return it
          // to `Pn` with the new minimum
          // update tuple, set new min and flip
          std::swap(std::get<Pn>(*it),std::get<Pc>(*it));
          std::get<Pn>(*it) = min_pn;
        }
        prev_pn = next_pn;
      }

      if (!found_flip) {
        // TODO: don't do this for the first or last bucket...
        // TODO: we need only one flipped per bucket
        tuple_t t = *eqr.first;
        std::swap(std::get<Pn>(t),std::get<Pc>(t));
        newtuples.push_back(t);
      }

      // next range
      begin = eqr.second;
    }


    MP_TIMER_END_SECTION("local flips");
    std::size_t nnew = newtuples.size();
    // insert at end and swap into active part of vector
    std::size_t active_size = pend - localVector.begin();
    std::size_t inactive_size = localVector.end() - pend;
    localVector.insert(localVector.end(), newtuples.begin(), newtuples.end());
    std::size_t nswap = std::min(nnew, inactive_size);
    for (std::size_t i = 0; i < nswap; ++i) {
        std::swap(localVector[active_size+i], localVector[localVector.size()-1-i]);
    }
    pend = localVector.begin() + active_size + nnew;
    MP_TIMER_END_SECTION("vector inserts");

    pend = std::partition(localVector.begin(), pend, [](tuple_t& t){return std::get<Pn>(t) != inactive_partition;});
    MP_TIMER_END_SECTION("std::partition");

    // load balance
    if (load_balance)
      pend = mxx::block_decompose_partitions(localVector.begin(), pend, localVector.end(), comm);

    // check if all processors are done
    keepGoing = !mxx::test_all(done);
    MP_TIMER_END_SECTION("check termination");

    countIterations++;
    if(!rank)
      std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";

    std::stringstream ss;
    ss << "logSort.g500.iter_" << countIterations;
//    dump_vector(localVector, comm, ss.str());
  }

  double time = t.elapsed() - startTime;


    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }
}



#endif
