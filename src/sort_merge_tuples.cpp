//Includes
#include <mpi.h>
#include <iostream> 

//from external repository
#include "timer.hpp"


//File includes from BLISS
#include "common/kmer.hpp"
#include "common/base_types.hpp"

//Own includes
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"


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

int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  //Specify the fileName
  std::string filename; 
  if( argc == 2 ) {
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
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank)
  {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }
  
  timer t;
  double start = t.get_ms();

  //Initialize the KmerVector
  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;
  std::vector<tuple_t> localVector;

  MP_TIMER_START();

  //Populate localVector for each rank and return the vector with all the tuples
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector); 

  MP_TIMER_END_SECTION("Read data from disk");


  //Sort tuples by KmerId
  char keepGoing = 1;
  int countIterations = 0;
  //keepGoing will be updated here
  char localKeepGoing;

  while(keepGoing)
  {
	  {
    MP_TIMER_START();
    //Sort by Kmers
    //Update P_n
    sortTuples<0,1,false> (localVector);
    MP_TIMER_END_SECTION("iteration KMER phase completed");
	  }

	  localKeepGoing = true;
	  {
    MP_TIMER_START();


    //Sort by P_c
    //Update P_n and P_c both
    sortTuples<2,1,true> (localVector, localKeepGoing);
    MP_TIMER_END_SECTION("iteration PARTITION phase completed");
	  }

	  {
    MP_TIMER_START();

    //Check whether all processors are done
    MPI_Allreduce(&localKeepGoing, &keepGoing, 1, MPI_CHAR , MPI_MAX, MPI_COMM_WORLD);
    MP_TIMER_END_SECTION("iteration Check phase completed");
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

  double time = t.get_ms() - start;
  if(!rank)
  {
    std::cout << "Algorithm took " << countIterations << " iteration.\n"; 
    std::cout << "TOTAL TIME : " << time << " ms.\n"; 
  }


  MPI_Finalize();   
  return(0);

}


