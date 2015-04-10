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

//from external repository
#include <timer.hpp>

#include <sstream>



int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  //Specify the fileName
  std::string filename;
  if( argc > 1 ) {
    filename = argv[1];
  }
  else {
    std::cout << "Usage: mpirun -np 4 <executable> FASTQ_FILE\n";
    return 1;
  }

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
  typedef PartitionReduceAndMarkAsInactive<2, 1, tuple_t> PartitionReducerType;
  ActivePartitionPredicate<1, tuple_t> app;

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
  MP_TIMER_START();
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



  while (keepGoing) {
    mxx::sort(localVector.begin(), localVector.end(), // TODO: only active partitions
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<2>(x) < std::get<2>(x)
            || (std::get<2>(x) == std::get<2>(y)
            && std::get<1>(x) < std::get<1>(y));
        });
    layer_comparator<2> pc_comp;

    // scan and find new p
    auto begin = localVector.begin();
    auto end = localVector.end();
    bool done = true;
    // for each range/bucket
    for(; begin != end; ) {
      // find bucket (linearly)
      auto eqr = findRange(begin, end,*begin, pc_comp);
      auto min = *eqr.first;

      if (eqr.first + 1 == eqr.second) {
        // single element -> change partition to smaller
        done = false;
        // next range
        begin = eqr.second;
        continue;
      }
      // TODO: compress the done elements!
      // check if all identical, then done!
      if (std::get<1>(*eqr.first) == std::get<1>(*(eqr.second-1))) {
          for (auto it = eqr.first; it !+ eqr.second; ++it) {
            std::get<2>(*it) = std::get<1>(*it);
          }
          continue;
      }

      done = false;

      // for each further element:
      bool found_flip = false;
      for (auto it = eqr.first+1; it !+ eqr.second; ++it) {
        if (std::get<1>(*it) == std::get<2>(*it)) {
          if (!found_flip) {
            std::cout << "Found eq (" << std::get<1>(*it) << "," << std::get<2>(*it) << ")" << std::endl;
            found_flip = true;
            std::get<1>(*it) = std::get<2>(*it);
            std::get<2>(*it) = min;
          } else {
            // moves over to new partition
            std::get<1>(*it) = min;
            std::get<2>(*it) = min;
          }
        } else if (std::get<1>(*it) > std::get<2>(*it)) {
          // found flipped, set min and unflip
          std::swap(std::get<1>(*it),std::get<2>(*it));
          std::get<1>(*it) = min;
        } else {
          // update tuple, set new min and flip
          if (!found_flip && std::get<2>(*(it-1)) == std::get<1>(*it)) {
            // set flipped
            std::cout << "Found repeat (" << std::get<1>(*it) << "," << std::get<2>(*it) << ")" << std::endl;
            found_flip = true;
            std::get<1>(*it) = std::get<2>(*it);
            std::get<2>(*it) = min;
          } else {
            std::swap(std::get<1>(*it),std::get<2>(*it));
            std::get<1>(*it) = min;
          }
        }
      }

      if (!found_flip) {
        std::cout << "no flip found!" << std::endl;
        exit(EXIT_FAILURE);
      }

      // next range
      begin = eqr.second

    }
    /*
    // sort by (P_c,P_n) and update P_c via P_n
    mxx::sort(localVector.begin(), localVector.end(), // TODO: only active partitions
        [](const tuple_t& x, const tuple_t&y){
        return (std::get<2>(x) < std::get<2>(x)
            || (std::get<2>(x) == std::get<2>(y)
            && std::get<1>(x) < std::get<1>(y));
        }
        , MPI_COMM_WORLD, false);

    // reduce over tuple

    // sort by P_c and update P_c via P_n
    mxx::sort(start, pend, layer_comparator<2, tuple_t>(), MPI_COMM_WORLD, true);
    PartitionReducerType r2;
    r2(start, pend, MPI_COMM_WORLD);
    //sortAndReduceTuples<2, PartitionReducerType, tuple_t> (start, pend, MPI_COMM_WORLD);
    MP_TIMER_END_SECTION("iteration PARTITION phase completed");

    // check for global termination
    keepGoing = !checkTermination<1, tuple_t>(start, pend, MPI_COMM_WORLD);
    MP_TIMER_END_SECTION("iteration Check phase completed");

    if (keepGoing) {
      // now reduce to only working with active partitions
      pend = std::partition(start, pend, app);
      MP_TIMER_END_SECTION("iteration std::partition completed");
      // re-shuffle the partitions to counter-act the load-inbalance
      pend = mxx::block_decompose_partitions(start, pend, end);
      MP_TIMER_END_SECTION("iteration mxx::block_decompose_partitions completed");
    }
    */

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

  MPI_Finalize();
  return(0);
}


