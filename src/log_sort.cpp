/**
 * @file    log_sort.cpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the de bruijn graph clustering in log(D_max) time.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
//Includes
#include <mpi.h>
#include <iostream>

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"

#include <mxx/collective.hpp>
#include <mxx/distribution.hpp>
#include <mxx/reduction.hpp>

//from external repository
#include <timer.hpp>

#include <sstream>

// sequential log(D_max) implementation
void cluster_reads_seq(const std::string& filename)
{
  //Specify Kmer Type
  const int kmerLength = 31;
  typedef bliss::common::DNA AlphabetType;
  typedef bliss::common::Kmer<kmerLength, AlphabetType, uint64_t> KmerType;

  //Assuming kmer-length is less than 32
  typedef uint64_t KmerIdType;

  //Assuming read count is less than 4 Billion
  typedef uint32_t ReadIdType;

  // define k-mer and operator types
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }

  timer t;
  double startTime = t.get_ms();

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  //MP_TIMER_START();
  MP_TIMER_START();

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, MPI_COMM_WORLD);

  MP_TIMER_END_SECTION("Read data from disk");


  // re-distirbute vector into equal block partition
  //mxx::block_decompose(localVector);

  assert(localVector.size() > 0);
  auto start = localVector.begin();
  auto end = localVector.end();
  auto pend = end;


  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;
  // sort by k-mer is first step (and then never again done)

  // Pc == Pn

  // sort by k-mers and update Pn
  mxx::sort(start, pend, layer_comparator<0, tuple_t>(), MPI_COMM_WORLD, false);
  // order of template params:
  // template <uint8_t keyLayer, uint8_t reductionLayer, uint8_t resultLayer, typename T>
  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */
  typedef KmerReduceAndMarkAsInactive<0, 2, 1, tuple_t> KmerReducerType;
  KmerReducerType r1;
  r1(start, pend, MPI_COMM_WORLD);
  MP_TIMER_END_SECTION("iteration KMER phase completed");

  // gather everything (for sequential test)
  //std::vector<tuple_t> globalvec = mxx::gatherv(localVec);
  if (p != 1) {
    std::cerr << "works only for p=1" << std::endl;
    exit(EXIT_FAILURE);
  }


  for (auto it = localVector.begin(); it != localVector.end(); ++it)
  {
    if (std::get<1>(*it) >= std::numeric_limits<int>::max()-1)
      std::get<1>(*it) = std::get<2>(*it);
  }

  while (keepGoing) {
    std::sort(localVector.begin(), localVector.end(), // TODO: only active partitions
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<2>(x) < std::get<2>(y)
            || (std::get<2>(x) == std::get<2>(y)
            && std::get<1>(x) < std::get<1>(y)));
        });
    layer_comparator<2, tuple_t> pc_comp;

    // scan and find new p
    auto begin = localVector.begin();
    auto end = localVector.end();
    std::vector<tuple_t> newtuples;
    bool done = true;
    // for each range/bucket
    //
    // exchange bucket min/start !?
    //
    for(; begin != end; ) {
      // find bucket (linearly)
      auto eqr = findRange(begin, end, *begin, pc_comp);
      auto min = std::get<1>(*eqr.first);

      //int i = eqr.first - begin;
      //int j = eqr.second - begin;
      //assert(min != 676);

      if (eqr.first + 1 == eqr.second) {
        // single element, wait till paired up!
        //done = false;
        std::get<2>(*eqr.first) = std::get<1>(*eqr.first); // now useless!
        begin = eqr.second;
        continue;
      }
      // TODO: compress the done elements!
      // check if all identical, then done!
      if (std::get<1>(*eqr.first) == std::get<1>(*(eqr.second-1))) {
          for (auto it = eqr.first; it != eqr.second; ++it) {
            std::get<2>(*it) = std::get<1>(*it);
          }
          begin = eqr.second;
          continue;
      }

      done = false;

      // for each further element:
      bool found_flip = false;
      uint32_t prev_pn = min;
      for (auto it = eqr.first+1; it != eqr.second; ++it) {
        uint32_t next_pn = std::get<1>(*it);
        // if repeat or both entries equal:
        if (std::get<1>(*it) == prev_pn || std::get<1>(*it) == std::get<2>(*it)) {
          if (!found_flip) {
            // set flipped
            found_flip = true;
            std::get<1>(*it) = std::get<2>(*it);
            std::get<2>(*it) = min;
          } else {
            // moves over to new partition (since duplicates/equals are not needed)
            std::get<1>(*it) = min;
            std::get<2>(*it) = min;
          }
          // Pn > Pc (a flipped thing) -> unflip and return
        } else if (std::get<1>(*it) > std::get<2>(*it)) {
          std::swap(std::get<1>(*it),std::get<2>(*it));
          std::get<1>(*it) = min;
        } else {
          // Pc > Pn => new min
          // update tuple, set new min and flip
          std::swap(std::get<1>(*it),std::get<2>(*it));
          std::get<1>(*it) = min;
        }
        prev_pn = next_pn;
      }

      if (!found_flip) {
        tuple_t t = *eqr.first;
        std::swap(std::get<1>(t),std::get<2>(t));
        newtuples.push_back(t);
      }

      // next range
      begin = eqr.second;
    }
    keepGoing = !done;
    localVector.insert(localVector.end(), newtuples.begin(), newtuples.end());

    countIterations++;
    if(!rank)
      std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";
  }


#if OUTPUTTOFILE
  //Output all (Kmer, PartitionIds) to a file in sorted order by Kmer
  //Don't play with the 0, 2 order, this is assumed by outputCompare
  if(!rank) std::cout << "WARNING: write to file option enabled \n";
  writeTuplesAll<0, 2, tuple_t>(localVector.begin(), localVector.end(), filename);
#endif

  double time = t.get_ms() - startTime;


    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }
}

// parallel/MPI log(D_max) implementation
void cluster_reads_par(const std::string& filename)
{
  static constexpr int Pn = 1;
  static constexpr int Pc = 2;
  //Specify Kmer Type
  const int kmerLength = 31;
  typedef bliss::common::DNA AlphabetType;
  typedef bliss::common::Kmer<kmerLength, AlphabetType, uint64_t> KmerType;

  //Assuming kmer-length is less than 32
  typedef uint64_t KmerIdType;

  //Assuming read count is less than 4 Billion
  typedef uint32_t ReadIdType;

  // define k-mer and operator types
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }

  timer t;
  double startTime = t.get_ms();

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  MP_TIMER_START();

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, MPI_COMM_WORLD);

  MP_TIMER_END_SECTION("Read data from disk");

  assert(localVector.size() > 0);
  auto start = localVector.begin();
  auto end = localVector.end();
  auto pend = end;


  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;
  // sort by k-mer is first step (and then never again done)

  // sort by k-mers and update Pn
  mxx::sort(start, pend, layer_comparator<0, tuple_t>(), MPI_COMM_WORLD, false);
  typedef KmerReduceAndMarkAsInactive<0, Pc, Pn, tuple_t> KmerReducerType;
  KmerReducerType r1;
  r1(start, pend, MPI_COMM_WORLD);
  MP_TIMER_END_SECTION("iteration KMER phase completed");


  // remove the MAX thing after k-mer reduce:
  // TODO: own k-mer reduction function!
  for (auto it = localVector.begin(); it != localVector.end(); ++it)
  {
    if (std::get<Pn>(*it) >= std::numeric_limits<int>::max()-Pn)
      std::get<Pn>(*it) = std::get<Pc>(*it);
  }

  while (keepGoing) {
    mxx::sort(localVector.begin(), localVector.end(),
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<Pc>(x) < std::get<Pc>(y)
            || (std::get<Pc>(x) == std::get<Pc>(y)
            && std::get<Pn>(x) < std::get<Pn>(y)));
        }, MPI_COMM_WORLD, false);
    MP_TIMER_END_SECTION("mxx::sort");

    layer_comparator<Pc, tuple_t> pc_comp;

    // scan and find new p
    auto begin = localVector.begin();
    auto end = localVector.end();
    std::vector<tuple_t> newtuples;
    bool done = true;

    // find last bucket start and send across boundaries!
    tuple_t last_min = *(end-1);
    last_min = *std::lower_bound(begin, end, last_min, pc_comp);
    // for each processor, get the first element of the last most element
    tuple_t prev_min = mxx::exscan(last_min,
        [](const tuple_t& x, const tuple_t& y){
          // return max Pc, if equal, return min Pn
          if (std::get<Pc>(x) < std::get<Pc>(y) ||
            (std::get<Pc>(x) == std::get<Pc>(y)
             && std::get<Pn>(x) > std::get<Pn>(y)))
          return y;
          else return x;}
        , MPI_COMM_WORLD);
    tuple_t prev_el = mxx::right_shift(*(end-1), MPI_COMM_WORLD);

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
        , MPI_COMM_WORLD);
    //tuple_t next_el = mxx::left_shift(*begin, MPI_COMM_WORLD);

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
      uint32_t prev_pn = std::get<Pn>(prev_el);
      auto it = eqr.first;
      if (rank == 0 || (rank > 0 && std::get<Pc>(*eqr.first) != std::get<Pc>(prev_el))) {
          // skip first (since it is the min entry
          prev_pn = min_pn;
          it++;
      }
      for (; it != eqr.second; ++it) {
        uint32_t next_pn = std::get<Pn>(*it);
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


#if OUTPUTTOFILE
  //Output all (Kmer, PartitionIds) to a file in sorted order by Kmer
  //Don't play with the 0, 2 order, this is assumed by outputCompare
  if(!rank) std::cout << "WARNING: write to file option enabled \n";
  writeTuplesAll<0, Pc, tuple_t>(localVector.begin(), localVector.end(), filename);
#endif

  double time = t.get_ms() - startTime;


    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }
}

// parallel/MPI log(D_max) implementation with removal of inactive partitions
void cluster_reads_par_inactive(const std::string& filename, bool load_balance)
{
  static constexpr int Pn = 1;
  static constexpr int Pc = 2;
  static constexpr uint32_t inactive_partition = std::numeric_limits<uint32_t>::max();
  //Specify Kmer Type
  const int kmerLength = 31;
  typedef bliss::common::DNA AlphabetType;
  typedef bliss::common::Kmer<kmerLength, AlphabetType, uint64_t> KmerType;

  //Assuming kmer-length is less than 32
  typedef uint64_t KmerIdType;

  //Assuming read count is less than 4 Billion
  typedef uint32_t ReadIdType;

  // define k-mer and operator types
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n";
    std::cout << "Filename : " <<  filename << "\n";
  }

  timer t;
  double startTime = t.get_ms();

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  MP_TIMER_START();

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, MPI_COMM_WORLD);

  MP_TIMER_END_SECTION("Read data from disk");

  assert(localVector.size() > 0);
  auto start = localVector.begin();
  auto end = localVector.end();
  auto pend = end;

  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;


  // sort by k-mers and update Pn
  mxx::sort(start, pend, layer_comparator<0, tuple_t>(), MPI_COMM_WORLD, false);
  typedef KmerReduceAndMarkAsInactive<0, Pc, Pn, tuple_t> KmerReducerType;
  KmerReducerType r1;
  r1(start, pend, MPI_COMM_WORLD);
  MP_TIMER_END_SECTION("iteration KMER phase completed");

  // remove the MAX thing after k-mer reduce:
  // TODO: own k-mer reduction function!
  for (auto it = localVector.begin(); it != localVector.end(); ++it)
  {
    if (std::get<Pn>(*it) >= std::numeric_limits<int>::max()-Pn)
      std::get<Pn>(*it) = std::get<Pc>(*it);
  }

  while (keepGoing) {
    mxx::sort(localVector.begin(), pend,
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<Pc>(x) < std::get<Pc>(y)
            || (std::get<Pc>(x) == std::get<Pc>(y)
            && std::get<Pn>(x) < std::get<Pn>(y)));
        }, MPI_COMM_WORLD, false);
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
    std::vector<std::size_t> distr = mxx::allgather(local_size, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "local_sizes: " << distr << std::endl;
    }

    int color = local_size == 0 ? 0 : 1;
    MPI_Comm nonempty_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &nonempty_comm);
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
      uint32_t prev_pn = std::get<Pn>(prev_el);
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
        uint32_t next_pn = std::get<Pn>(*it);
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
      pend = mxx::block_decompose_partitions(localVector.begin(), pend, localVector.end(), MPI_COMM_WORLD);

    // check if all processors are done
    keepGoing = !mxx::test_all(done);
    MP_TIMER_END_SECTION("check termination");

    countIterations++;
    if(!rank)
      std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";
  }

#if OUTPUTTOFILE
  //Output all (Kmer, PartitionIds) to a file in sorted order by Kmer
  //Don't play with the 0, 2 order, this is assumed by outputCompare
  if(!rank) std::cout << "WARNING: write to file option enabled \n";
  writeTuplesAll<0, Pc, tuple_t>(localVector.begin(), localVector.end(), filename);
#endif

  double time = t.get_ms() - startTime;


    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }
}

int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  //Specify the fileName
  std::string filename;
  std::string method;
  if( argc > 2 ) {
    method = argv[1];
    filename = argv[2];
  }
  else {
    std::cout << "Usage: mpirun -np 4 <executable> <method> FASTQ_FILE\n";
    std::cout << "  where <method> can be: standard,inactive,loadbalance\n";
    return 1;
  }

  if (method == "standard")
    cluster_reads_par(filename);
  else if (method == "inactive")
    cluster_reads_par_inactive(filename, false);
  else if (method == "loadbalance")
    cluster_reads_par_inactive(filename, true);
  else {
    std::cout << "Usage: mpirun -np 4 <executable> <method> FASTQ_FILE\n";
    std::cout << "  where <method> can be: standard,inactive,loadbalance\n";
    return 1;
  }

  MPI_Finalize();
  return(0);
}


