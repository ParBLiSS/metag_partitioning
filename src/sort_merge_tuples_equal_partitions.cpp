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
  typedef KmerReduceAndMarkAsInactive<0, 2, 1, tuple_t> KmerReducerType;
  typedef PartitionReduceAndMarkAsInactive<2, 1, tuple_t> PartitionReducerType;

  // get communicaiton parameters
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

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
  mxx::block_decompose(localVector);

  assert(localVector.size() > 0);
  auto start = localVector.begin();
  auto end = localVector.end();
  auto pend = end;


  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;


  ActivePartitionPredicate<1, tuple_t> app;
  layer_comparator<0, tuple_t> kmer_comp;
  layer_comparator<2, tuple_t> pc_comp;
  KmerReducerType kmer_reduc;
  PartitionReducerType part_reduc;

  while (keepGoing) {
//    MP_TIMER_START();


    // sort by k-mer (only active k-mers)
    mxx::sort(start, pend, kmer_comp, comm, true);
    MP_TIMER_END_SECTION("mxx::sort by k-mer (active kmers)");

    // reduce through k-mer neighborhood
    //Update P_n
    kmer_reduc(start, pend, comm);
    MP_TIMER_END_SECTION("k-mer reduction (active kmers)");

    // sort by Pc (previous partition)
    mxx::sort(start, pend, pc_comp, comm, true);
    MP_TIMER_END_SECTION("mxx::sort by Pc (active kmers)");

    // update partition (Pn -> Pc)
    //Update P_n and P_c both
    part_reduc(start, pend, comm);
    MP_TIMER_END_SECTION("partition reduction Pn->Pc (all kmers)");


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


