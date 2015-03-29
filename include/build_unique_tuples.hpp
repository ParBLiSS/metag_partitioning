#ifndef BUILD_UNIQUE_TUPLES_HPP 
#define BUILD_UNIQUE_TUPLES_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp> 
#include <mxx/datatypes.hpp>
#include <mxx/shift.hpp>

//Specify mxx library what MPI_Datatype to use while dealing with kmers
//TODO: How to generalize this?
namespace mxx {
    template <>
    class datatype<bliss::common::Kmer<31, bliss::common::DNA, uint32_t>>: public datatype_contiguous<bliss::common::Kmer<31, bliss::common::DNA, uint32_t>::KmerWordType,  bliss::common::Kmer<31, bliss::common::DNA, uint32_t>::nWords> {};
}


/**
 * @details
 * Given a kmer, replicates it into tuple of three elements and inserts to the given vector
 *
 * @TODO: This method is hard coded assuming 64 bits are enough to uniquely identify every kmer 
 *        Need to generalise this later.
 */
template <typename KmerType, typename VectorTupleType>
void insertKmerAsTuple(KmerType& kmer, VectorTupleType& uniqueKmerTupleVector)
{
  //getPrefix() gives a 64-bit prefix for hashing assuming 
  //Assuming kmer size in bits is <= 64
  
  auto tupleToInsert = std::make_tuple(kmer, kmer.getPrefix(), kmer.getPrefix()); 
  uniqueKmerTupleVector.push_back(tupleToInsert);
}

/**
 * @details
 * Uses Patrick's sample sort to remove duplicate kmers
 *
 * @param[in] kmerVectorWithDuplicates  Vector of Kmers not necessarily unique
 * @param[out] uniqueKmerTupleVector    Vector of tuples (Kmer, P_id_new, P_id_current) and 
 *                                      each Kmer is unique. P_id_new and P_id_current will be 
 *                                      initialised to Kmer value in the beginning
 *
 * @note                                This function should be called by all MPI ranks
 */
template <typename KmerType, typename T, typename VectorTupleType>
void sortAndRemoveKmerDuplicates(T& kmerVectorWithDuplicates, VectorTupleType& uniqueKmerTupleVector,
                                MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;

  /// MPI rank within the communicator
  MPI_Comm_rank(comm, &rank);

  //Do the sort
  mxx::sort(kmerVectorWithDuplicates.begin(), kmerVectorWithDuplicates.end(), std::less<KmerType>(), MPI_COMM_WORLD, false);  

  //Remove duplicates
  //Before that we need 1 right shift permutation for communicating boundary kmer

  KmerType kmerReceived = mxx::right_shift(kmerVectorWithDuplicates.back());

  //Start iterating over vector and remove duplicates
  //While doing this we can construct tuples 
  
  //Need special treatment for the first iteration
  bool checkingFirstElement = true;
  KmerType previousKmerSeen;
  for ( auto& eachKmer : kmerVectorWithDuplicates) 
  {
    if(checkingFirstElement)
    {
      if(rank == 0)
        insertKmerAsTuple(eachKmer, uniqueKmerTupleVector); 
      else
      {
        if(eachKmer != kmerReceived)
          insertKmerAsTuple(eachKmer, uniqueKmerTupleVector); 
      }
    }
    else
    {
      if(eachKmer != previousKmerSeen)
        insertKmerAsTuple(eachKmer, uniqueKmerTupleVector); 
    }

    //Store this kmer for next iteration
    previousKmerSeen = eachKmer;
    checkingFirstElement = false;
  }

  std::cerr << "[sortAndRemoveKmerDuplicates] Rank : " << rank << " has " << uniqueKmerTupleVector.size() << " tuples after sort and keeping uniques." << std::endl; 
}


#endif
