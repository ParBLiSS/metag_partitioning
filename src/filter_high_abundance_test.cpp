//Includes
#include <mpi.h>
#include <iostream>

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "parallel_fastq_iterate.hpp"
#include "preProcess.hpp"

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

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }

  timer t;
  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, MPI_COMM_WORLD);


  // re-distirbute vector into equal block partition
  assert(localVector.size() > 0);

  trimHighFrequencyKmer<0, 1>(localVector);


  MPI_Finalize();
  return(0);
}


