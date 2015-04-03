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
#include <timer.hpp>

#include <sstream>

/*
 * Uses timer.hpp from Patrick's psack-copy
 * Couple of variables like p and rank should be defined as communicator size and MPI rank within the code
 */

#define MP_ENABLE_TIMER 1
#if MP_ENABLE_TIMER
#define MP_TIMER_START() TIMER_START()
#define MP_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#else
#define MP_TIMER_START()
#define MP_TIMER_END_SECTION(str)
#endif

//To output all the kmers and their respective partitionIds
//Switch on while testing
#define OUTPUTTOFILE 0


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

  typedef uint32_t activeFlagType;

  //Know rank
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank)
  {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }
  

  //Initialize the KmerVector
  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   * 3 : active/inactive partition at bit 1, interior/boundary kmer at bit 0.
   */
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType, activeFlagType> tuple_t;
  std::vector<tuple_t> localVector;

  typedef KmerReduceAndMarkAsInactive<0, 2, 1, 3, tuple_t> KmerReducerType;
  typedef PartitionReduceAndMarkAsInactive<2, 1, 3, tuple_t> PartitionReducerType;

  MP_TIMER_START();

  //Populate localVector for each rank and return the vector with all the tuples
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, MPI_COMM_WORLD);

  MP_TIMER_END_SECTION("Read data from disk");

//  printTuples<0, 2, tuple_t>(localVector.begin(), localVector.end(), MPI_COMM_WORLD);

  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;

  auto start = localVector.begin();
  auto end = localVector.end();

  ActivePartitionPredicate<3, tuple_t> app;


    while (keepGoing)
    {
    	{
      MP_TIMER_START();
      //Sort by Kmers
      //Update P_n
      sortAndReduceTuples<0, KmerReducerType, tuple_t> (start, end, MPI_COMM_WORLD);
      MP_TIMER_END_SECTION("iteration KMER phase completed");
    	}

    	{
      MP_TIMER_START();
      //Sort by P_c
      //Update P_n and P_c both
      sortAndReduceTuples<2, PartitionReducerType, tuple_t> (start, end, MPI_COMM_WORLD);
      MP_TIMER_END_SECTION("iteration PARTITION phase completed");
    	}

    	{
      MP_TIMER_START();
      //Check whether all processors are done
      keepGoing = !checkTermination<3, tuple_t>(start, end, MPI_COMM_WORLD);
      MP_TIMER_END_SECTION("iteration Check phase completed");
    	}


    	// now reduce to only working with active partitions
    	end = std::partition(start, end, app);
//    	std::sort(start, end, layer_comparator<2, tuple_t>)


      countIterations++;
      if(!rank)
        std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";
    }


#ifdef OUTPUTTOFILE
  //Output all (Kmer, PartitionIds) to a file in sorted order by Kmer
  //Don't play with the 0, 2 order, this is assumed by outputCompare
  std::cout << "WARNING: write to file option enabled \n";
  writeTuplesAll<0, 2, tuple_t>(localVector.begin(), localVector.end(), filename);
#endif



    if(!rank)
      std::cout << "Algorithm took " << countIterations << " iteration.\n";

  MPI_Finalize();   
  return(0);
}


