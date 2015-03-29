#ifndef BUILD_UNIQUE_TUPLES_HPP 
#define BUILD_UNIQUE_TUPLES_HPP 

//Includes
#include <mpi.h>

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
void sortAndRemoveKmerDuplicates(T& kmerVectorWithDuplicates, VectorTupleType& uniqueKmerTupleVector)
{

}


#endif
