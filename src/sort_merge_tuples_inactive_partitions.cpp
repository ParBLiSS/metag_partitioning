//Includes
#include <mpi.h>
#include <iostream> 

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"

//from external repository
#include <mxx/timer.hpp>

#include <sstream>

/*
 * Uses timer.hpp from Patrick's psack-copy
 * Couple of variables like p and rank should be defined as communicator size and MPI rank within the code
 */

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

  //Know rank
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  if(!rank)
  {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }

  mxx::timer t;
  double startTime = t.elapsed();

  //Initialize the KmerVector
  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;
  std::vector<tuple_t> localVector;

  typedef KmerReduceAndMarkAsInactive<0, 2, 1, tuple_t> KmerReducerType;
  typedef PartitionReduceAndMarkAsInactive<2, 1, tuple_t> PartitionReducerType;

  MP_TIMER_START();

  //Populate localVector for each rank and return the vector with all the tuples
  std::vector<bool> readFilterFlags;
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, readFilterFlags, MPI_COMM_WORLD);

  MP_TIMER_END_SECTION("Read data from disk");

  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;

  assert(localVector.size() > 0);

  auto start = localVector.begin();
  auto pend = localVector.end();

  ActivePartitionPredicate<1, tuple_t> app;
  layer_comparator<0, tuple_t> kmer_comp;
  layer_comparator<2, tuple_t> pc_comp;
  KmerReducerType kmer_reduc;
  PartitionReducerType part_reduc;

    while (keepGoing)
    {

      // sort by k-mer (only active k-mers)
      mxx::sort(start, pend, kmer_comp, comm, false);
      MP_TIMER_END_SECTION("mxx::sort by k-mer (active kmers)");

      // reduce through k-mer neighborhood
      //Update P_n
      kmer_reduc(start, pend, comm);
      MP_TIMER_END_SECTION("k-mer reduction (active kmers)");

      // sort by Pc (previous partition)
      mxx::sort(start, pend, pc_comp, comm, false);
      MP_TIMER_END_SECTION("mxx::sort by Pc (active kmers)");

      // update partition (Pn -> Pc)
      //Update P_n and P_c both
      part_reduc(start, pend, comm);
      MP_TIMER_END_SECTION("partition reduction Pn->Pc (all kmers)");


      // after this step, 1 Pc may be split up across processors.
      // but the partition sort in the next step will bring them back together.
      // since the interior kmers are moved as well, all tuples in a partition are still contiguous.

      //Check whether all processors are done
      keepGoing = !checkTermination<1, tuple_t>(start, pend, MPI_COMM_WORLD);
      MP_TIMER_END_SECTION("iteration Check phase completed");

      if (keepGoing) {
        // now reduce to only working with active partitions
        pend = std::partition(start, pend, app);
        MP_TIMER_END_SECTION("std::partition partitions (all kmers)");
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


    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }

  MPI_Finalize();   
  return(0);
}


