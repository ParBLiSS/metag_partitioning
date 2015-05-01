//Includes
#include <mpi.h>


//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "parallel_fastq_iterate.hpp"

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

  //Assuming read cout is less than 4 Billion
  typedef uint32_t ReadIdType;


  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;
  std::vector<tuple_t> localVector;

  //Populate localVector for each rank
  std::vector<bool> readFilterFlags;
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, readFilterFlags, MPI_COMM_WORLD);

  MPI_Finalize();   
  return(0);
}

