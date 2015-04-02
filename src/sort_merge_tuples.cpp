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

  while(keepGoing)
  {
    MP_TIMER_START();
    //Sort by Kmers
    //Update P_n
    sortTuples<0,1,false> (localVector);

    //keepGoing will be updated here
    char localKeepGoing;

    //Sort by P_c
    //Update P_n and P_c both
    sortTuples<2,1,true> (localVector, localKeepGoing);

    //Check whether all processors are done
    MPI_Allreduce(&localKeepGoing, &keepGoing, 1, MPI_CHAR , MPI_MAX, MPI_COMM_WORLD);
    countIterations++;
    if(!rank)
      std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";

    MP_TIMER_END_SECTION("Partitioning iteration completed");
  }

  //printTuples(localVector);
  std::string ofname = filename;
  std::stringstream ss;
  ss << "." << rank << ".out";
  ofname.append(ss.str());

  writeTuples<0, 2>(localVector, ofname);


  if(!rank)
    std::cout << "Algorithm took " << countIterations << " iteration.\n"; 

  MPI_Finalize();   
  return(0);

}


