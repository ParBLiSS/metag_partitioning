#ifndef POST_PROCESS_HPP 
#define POST_PROCESS_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp>

//Own includes
#include "sortTuples.hpp"
#include "configParam.hpp"
#include "packedRead.hpp"

/*
 * @brief     Takes the kmer-pid tuples as input and spits out read-pid tuples as output
 * @details   
 *            1. Parse the file again to generate first_kmer-MAX-readId tuples and append to original vector
 *            2. Sort the vector by kmers and for each MAX tuple, append the new tuples with readId-Pid to a new vector
 *            3. The format of new tuples should be emptyread-readid-pid-0
 *            4. delete the original localVector, and return the new tuples 
 */
template <typename KmerType, typename T, typename Q>
void generateReadToPartitionMapping(const std::string& filename, 
                                  std::vector<T>& localVector, std::vector<Q>& newlocalVector, 
                                  std::vector<bool>& readFilterFlags,
                                  MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  //temporary vector for appending reads (first kmers) to localVector
  std::vector<T> localVector2;

  //Parse the first kmers of each read and keep in localVector2
  readFASTQFile< KmerType, includeFirstKmersinFilteredReads<KmerType> > (filename, localVector2, readFilterFlags);

  //Paste all the values into original vector
  localVector.insert(localVector.end(), localVector2.begin(), localVector2.end());
  localVector2.clear();

  static layer_comparator<kmerTuple::kmer, T> kmerCmp;

  //Sort by kmer Layer
  mxx::sort(localVector.begin(), localVector.end(), kmerCmp, comm, true); 

  //First resolve the first and last buckets
  //Vector for containing tuples with pids of leftmost and rightmost buckets
  std::vector<T> kmerIdsToSend(2);
  std::vector<T> kmerIdsReceived(2);

  //Disable the effect of these entries by default by setting MAX in the PnLayer 
  //Rationale: This rank might not contain valid Pid for boundary buckets
  std::get<kmerTuple::Pn>(kmerIdsToSend[0]) = MAX;  std::get<kmerTuple::Pn>(kmerIdsToSend[1]) = MAX;
  

  //First bucket's range
  auto firstBucketBound = findRange(localVector.begin(), localVector.end(), *(localVector.begin()), kmerCmp);

  for(auto it = firstBucketBound.first; it != firstBucketBound.second; it++)
  {
    //If there is a tuple containing the valid partition id for this kmer
    if(std::get<kmerTuple::Pn>(*it) != MAX)
    {
      //Save this tuple
      kmerIdsToSend[0] = *it;
    }
  }

  //Repeat the same process with rightmost bucket
  auto lastBucketBound = findRange(localVector.rbegin(), localVector.rend(), *(localVector.rbegin()), kmerCmp);
  for(auto it = lastBucketBound.first; it != lastBucketBound.second; it++)
  {
    //If there is a tuple containing the valid partition id for this kmer
    if(std::get<kmerTuple::Pn>(*it) != MAX)
    {
      //Save this tuple
      kmerIdsToSend[1] = *it;
    }
  }

  //Do the communication and get the partitionids for boundary partitions
  //We do the MPI exscan on rightMostKmersIdSend to get the ids for our bucket, higher rank 

  std::vector<T> gatherkmerIds = mxx::allgatherv(kmerIdsToSend, comm);

  //Fetch the correct leftmost and rightmost ids using the values gathered above
  kmerIdsReceived[0] = *(std::find_if(gatherkmerIds.begin(), gatherkmerIds.end(), 
                                                                              [&localVector = localVector](const T& x){
                                                                                return (std::get<kmerTuple::kmer>(x) == std::get<kmerTuple::kmer>(localVector.front()) &&
                                                                                        std::get<kmerTuple::Pn>(x) != MAX);}));
  kmerIdsReceived[1] = *(std::find_if(gatherkmerIds.begin(), gatherkmerIds.end(), 
                                                                              [&localVector = localVector](const T& x){
                                                                                return (std::get<kmerTuple::kmer>(x) == std::get<kmerTuple::kmer>(localVector.back()) &&
                                                                                        std::get<kmerTuple::Pn>(x) != MAX);}));

  //Now we are ready to go through each bucket
  //For each bucket, we first find the correct pid, then for each read tuple, make new tuple in newlocalVector with appropriate values

  for(auto it = localVector.begin(); it != localVector.end();)
  {
    auto innerLoopBound = findRange(it, localVector.end(), *it, kmerCmp);
    
    if (innerLoopBound.first == localVector.begin()) // first bucket
    {
      //Get the pid from boundary tuple
      auto correctPid = std::get<kmerTuple::Pc>(kmerIdsReceived[0]);

      //Scan the bucket
      for(auto it2 = innerLoopBound.first; it2!= innerLoopBound.second; it2++)
      {
        if(std::get<kmerTuple::Pn>(*it2) == MAX)
        {
          Q newTuple;
          std::get<readTuple::pid>(newTuple) = correctPid;
          std::get<readTuple::rid>(newTuple) = std::get<kmerTuple::Pc>(*it2);
          std::get<readTuple::cnt>(newTuple) = 0;

          //Push this tuple to new vector
          newlocalVector.push_back(newTuple);
        }
      }
    }
    else if (innerLoopBound.second == localVector.end()) // last bucket
    {
      //Get the pid from boundary tuple
      auto correctPid = std::get<kmerTuple::Pc>(kmerIdsReceived[1]);

      //Scan the bucket
      for(auto it2 = innerLoopBound.first; it2!= innerLoopBound.second; it2++)
      {
        if(std::get<kmerTuple::Pn>(*it2) == MAX)
        {
          Q newTuple;
          std::get<readTuple::pid>(newTuple) = correctPid;
          std::get<readTuple::rid>(newTuple) = std::get<kmerTuple::Pc>(*it2);
          std::get<readTuple::cnt>(newTuple) = 0;

          //Push this tuple to new vector
          newlocalVector.push_back(newTuple);
        }
      }
    }
    else if (innerLoopBound.first != localVector.begin() && innerLoopBound.second != localVector.end()) // middle bucket
    {
      //Get the pid from boundary tuple
      auto findTuple = *(std::find_if(innerLoopBound.first, innerLoopBound.second, 
                                                                              [](const T& x){
                                                                                return (std::get<kmerTuple::Pn>(x) != MAX);}));

      auto correctPid = std::get<kmerTuple::Pc>(findTuple);

      //Scan the bucket
      for(auto it2 = innerLoopBound.first; it2!= innerLoopBound.second; it2++)
      {
        if(std::get<kmerTuple::Pn>(*it2) == MAX)
        {
          Q newTuple;
          std::get<readTuple::pid>(newTuple) = correctPid;
          std::get<readTuple::rid>(newTuple) = std::get<kmerTuple::Pc>(*it2);
          std::get<readTuple::cnt>(newTuple) = 0;

          //Push this tuple to new vector
          newlocalVector.push_back(newTuple);
        }
      }
    }

    //Move to next bucket
    it = innerLoopBound.second;
  }
}

/*
 * @brief     Given readid-pid as input, this generates vector of vector with sequences belonging to a single partition
 * @details
 *            1. Parse all the unfiltered reads and append new tuples for them, with pid set to MAX in the beginning
 *            2. Sort the vector by readid, and assign pid to reads
 *            3. Sort the vector by pid and write adjust the boundary so that no 2 partitions spawn across processors
 *            4. Convert this vector to a vector of vector of sequences belonging to the same partition
 */
template <typename KmerType, typename Q> 
void generateSequencesVector(const std::string& filename,
                             std::vector<Q>& newLocalVector, std::vector<bool>& readFilterFlags,
                             MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  //temporary vector for appending reads (first kmers) to localVector
  std::vector<Q> newLocalVector2;

  //Parse the first kmers of each read and keep in localVector2
  readFASTQFile< KmerType, includeWholeReadinFilteredReads<KmerType> > (filename, newLocalVector2, readFilterFlags);

  //Paste all the values into original vector
  newLocalVector.insert(newLocalVector.end(), newLocalVector2.begin(), newLocalVector2.end());
  newLocalVector2.clear();

  static layer_comparator<readTuple::rid, Q> ridCmp;

  //Sort by read id Layer
  mxx::sort(newLocalVector.begin(), newLocalVector.end(), ridCmp, comm, true); 

  //Now each bucket will have only 2 elements, read tuple with pid and read tuple with sequence.
  //Goal is to place the sequence in the tuple with correct pid and later remove tuples without pid
  //Since a bucket could be spilled across processors, we will do a left shift of leftmost element, and push_back it to local vector
  //Then each bucket is processed only when its size is 2

  Q tupleToShift = newLocalVector.front();
  Q tupleFromShift = mxx::left_shift(tupleToShift, comm);

  //Last but one processor
  if(rank < p - 1) 
    newLocalVector.push_back(tupleFromShift);

  //Start processing the buckets
  for(auto it = newLocalVector.begin(); it != newLocalVector.end();)
  {
    auto innerLoopBound = findRange(it, newLocalVector.end(), *it, ridCmp);

    if(innerLoopBound.second - innerLoopBound.first == 2)
    {
      //Get the pid from boundary tuple
      auto findTupleWithSequence = *(std::find_if(innerLoopBound.first, innerLoopBound.second, 
            [](const Q& x){
            return (std::get<readTuple::pid>(x) == MAX);}));

      auto findTupleWithPid = *(std::find_if(innerLoopBound.first, innerLoopBound.second,
            [](const Q& x){
            return (std::get<readTuple::cnt>(x) == 0);}));

      //Update sequence and count
      std::get<readTuple::seq>(findTupleWithPid) = std::get<readTuple::seq>(findTupleWithSequence);
      std::get<readTuple::cnt>(findTupleWithPid) = std::get<readTuple::cnt>(findTupleWithSequence);

    }
    else
    {
      //This tuple is not needed, therefore should be removed later
      std::get<readTuple::pid>(*innerLoopBound.first) = MAX;
    }

    //Move to next bucket
    it = innerLoopBound.second;
  }

  //Now we can get rid of all the tuples who have pid as MAX
  auto cend = std::partition(newLocalVector.begin(), newLocalVector.end(), 
      [](const Q& x){
      return std::get<readTuple::pid>(x) != MAX;});

  newLocalVector.erase(cend, newLocalVector.end());

}

/*
 * @brief     With read sequences saved using vector of vector of strings, run parallel assembly using a assembler
 * @details
 *            1. Iterate over the element of vectors and dump the reads belonging to same partition into fasta file
 *            2. Run assembler and append the contigs to a file local to this processor
 *            3. Merge all the contig files in the end
 */
void runParallelAssembly();

//Wrapper for all the post processing functions
template <typename KmerType,  typename T>
void finalPostProcessing(std::vector<T>& localVector, std::vector<bool>& readFilterFlags, const std::string& filename)
{
  //Determining storage container for read sequences
  typedef readStorageInfo<typename KmerType::KmerAlphabet, typename KmerType::KmerWordType> ReadSeqTypeInfo; 

  //Type of array to hold read sequence
  typedef std::array<typename ReadSeqTypeInfo::ReadWordType, ReadSeqTypeInfo::nWords> ReadSeqType;

  //tuple of type <Sequence, ReadId, PartitionId, Size of read> 
  //This order is defined in configParam.hpp as <seq,rid,pid,cnt>
  typedef std::tuple<ReadSeqType, ReadIdType, ReadIdType, uint32_t> tuple_t;

  //New vector type needs to be defined to hold read sequences
  std::vector<tuple_t> newlocalVector;

  //Get the newlocalVector populated with readId and partitionIds
  generateReadToPartitionMapping<KmerType>(filename, localVector, newlocalVector, readFilterFlags);

  //Get the newlocalVector poulated with vector of vector of read strings belonging to the same partition
  generateSequencesVector<KmerType>(filename, newlocalVector, readFilterFlags);

}

#endif
