/*
 * Code takes a fastq file as input and generates the input of vector of tuples for partitioning method
 */

//Includes
#include <mpi.h>

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "parallel_fastq_iterate.hpp"
#include "build_unique_tuples.hpp"

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
  typedef bliss::common::Kmer<kmerLength, AlphabetType, uint32_t> KmerType;

  //Initialize the KmerVector
  std::vector<KmerType> localVector;

  //Populate localVector for each rank
  generateKmerVector<KmerType, AlphabetType> (filename, localVector); 

  std::vector<std::tuple<KmerType, uint64_t, uint64_t>> uniqueKmerTuples; 

  sortAndRemoveKmerDuplicates<KmerType>(localVector, uniqueKmerTuples);

  MPI_Finalize();   
  return(0);
}


