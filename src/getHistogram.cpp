//Includes
#include <mpi.h>
#include <iostream>

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "configParam.hpp"
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"
#include "utils.hpp"
#include "preProcess.hpp"
#include "postProcess.hpp"

#include <mxx/collective.hpp>
#include <mxx/distribution.hpp>
#include <mxx/timer.hpp>

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

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }

  /*
   * PREPROCESSING PHASE
   */
  //Specify Kmer Type
  const int kmerLength_pre = KMER_LEN_PRE;
  typedef bliss::common::DNA AlphabetType;
  typedef bliss::common::Kmer<kmerLength_pre, AlphabetType, uint64_t> KmerType_pre;

  //Assuming kmer-length is less than 32
  typedef uint64_t KmerIdType;

  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;
  std::vector<bool> readFilterFlags;

  //Generate kmer tuples, keep filter off
  readFASTQFile< KmerType_pre, includeAllKmers<KmerType_pre> > (filename, localVector, readFilterFlags);

  //Pre-process
  MP_TIMER_START();
  trimReadswithHighMedianOrMaxCoverage<>(localVector, readFilterFlags);
  MP_TIMER_END_SECTION("Digital normalization completed");

  //Delete the local vector
  localVector.resize(0);
  

  //Specify Kmer Type
  const int kmerLength = KMER_LEN;
  typedef bliss::common::Kmer<kmerLength, AlphabetType, uint64_t> KmerType;


  // define k-mer and operator types
  typedef KmerReduceAndMarkAsInactive<tuple_t> KmerReducerType;
  typedef PartitionReduceAndMarkAsInactive<tuple_t> PartitionReducerType;
  ActivePartitionPredicate<tuple_t> app;


  mxx::timer t;
  double startTime = t.elapsed();

  /*
   * IMPORTANT NOTE
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  // Populate localVector for each rank and return the vector with all the tuples
  readFASTQFile< KmerType, includeAllKmersinFilteredReads<KmerType> > (filename, localVector, readFilterFlags);
  readFilterFlags.clear();


  // re-distirbute vector into equal block partition
  mxx::block_decompose(localVector);

  assert(localVector.size() > 0);
  auto start = localVector.begin();
  auto end = localVector.end();
  auto pend = end;


  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;
  while (keepGoing) {

    // sort by k-mers and update Pn
    mxx::sort(start, pend, layer_comparator<kmerTuple::kmer, tuple_t>(), MPI_COMM_WORLD, true);
    KmerReducerType r1;
    r1(start, pend, MPI_COMM_WORLD);

    // sort by P_c and update P_c via P_n
    mxx::sort(start, pend, layer_comparator<kmerTuple::Pc, tuple_t>(), MPI_COMM_WORLD, true);
    PartitionReducerType r2;
    r2(start, pend, MPI_COMM_WORLD);

    // check for global termination
    keepGoing = !checkTermination<kmerTuple::Pn, tuple_t>(start, pend, MPI_COMM_WORLD);

    if (keepGoing) {
      // now reduce to only working with active partitions
      pend = std::partition(start, pend, app);
      // re-shuffle the partitions to counter-act the load-inbalance
      pend = mxx::block_decompose_partitions(start, pend, end);
    }

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

  double time = t.elapsed() - startTime;

  std::string histFileName = filename;
  histFileName += ".hist";

  if(!rank)
  {
    std::cout << "Algorithm took " << countIterations << " iteration.\n";
    std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    std::cout << "Generating histogram in file " << histFileName << "\n";
  }

  generatePartitionSizeHistogram<2>(localVector, histFileName);

  //Before beginning the post processing, we need to make sure Pn and Pc are equal for every tuple

  std::for_each(localVector.begin(), localVector.end(), [](tuple_t &t){ std::get<kmerTuple::Pn>(t) = std::get<kmerTuple::Pc>(t);});

  finalPostProcessing<KmerType>(localVector, readFilterFlags, filename);

  MPI_Finalize();
  return(0);
}


