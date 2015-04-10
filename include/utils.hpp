#ifndef METAG_UTILS_HPP 
#define METAG_UTILS_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp>

//Own includes
#include "sortTuples.hpp"
#include "prettyprint.hpp"

#include <fstream>
#include <iostream>


//Helper function for generating histogram
//Inserts the given value to the vector
template <typename T1, typename T2>
void insertToHistogram(std::vector<T1>& hist, T2 value)
{
  if(hist.size() < value + 1)
    hist.resize(value + 1);
  hist[value]++;
}

/**
 * @brief Generates a histogram to see partition size v/s count information
 * @tparam keyLayer   Should denote the partition_Id layer
 * @param start,end   Indicate the start and end iterator for local vector of 
 *                    tuples
 * @NOTE              Returns histogram on root only
 */
template <uint8_t keyLayer , typename T>
void generatePartitionSizeHistogram(typename std::vector<T> localVector, std::string filename, MPI_Comm comm = MPI_COMM_WORLD)
{
  /*
   * Approach :
   * 1. Do a global sort by keyLayer (Partition id).
   * 2. Compute the information Partition_Id:Size for boundary partitions (Avoid duplication).
   * 3. Do an all_gather of boundary values plugged with ranks.
   * 4. All boundary partitions should be owned by minimum rank that shares it.
   * 5. Locally compute the largest partition size and do MPI_Allreduce 
   *    with MPI_MAX.
   * 6. Build the histogram locally. 
   * 7. Finally do MPI_Reduce over whole histogram to root node.
   * 8. Write the output to file
   */

  int p;
  int rank;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &rank);

  static layer_comparator<keyLayer, T> pccomp;

  //Sort the vector by each tuple's keyLayer element
  mxx::sort(localVector.begin(), localVector.end(), pccomp, comm, false);

  //Iterate over tuples to compute boundary information
  
  bool iownLeftBucket, iownRightBucket, onlySingleLocalPartition;
  //Type of tuple to communicate : (Partition Id, rank, size)
  typedef std::tuple<uint32_t, int, uint64_t> tupletypeforbucketSize;
  std::vector<tupletypeforbucketSize> toSend;
  toSend.resize(2);

  //Find the left most bucket
  auto leftBucketRange = findRange(localVector.begin(), localVector.end(), *(localVector.begin()), pccomp);
  std::get<0>(toSend[0]) = std::get<keyLayer>(*localVector.begin()); 
  std::get<1>(toSend[0]) = rank;
  std::get<2>(toSend[0]) = leftBucketRange.second - leftBucketRange.first;

  //Find the right most bucket
  auto rightBucketRange = findRange(localVector.rbegin(), localVector.rend(), *(localVector.rbegin()), pccomp);
  std::get<0>(toSend[1]) = std::get<keyLayer>(*localVector.rbegin()); 
  std::get<1>(toSend[1]) = rank;
  std::get<2>(toSend[1]) = rightBucketRange.second - rightBucketRange.first;

  //If we have only single partition, make sure we are not creating duplicates
  if(std::get<0>(toSend[0]) == std::get<0>(toSend[1]))
  {
    //Make second send element's size zero
    std::get<2>(toSend[1]) = 0;
    onlySingleLocalPartition = true;
  }
  else
    onlySingleLocalPartition = false;

  //Gather all the boundary information
  auto allBoundaryPartitionSizes = mxx::allgather_vectors(toSend, comm);

  uint64_t leftBucketSize = 0;
  uint64_t rightBucketSize = 0;

  //Need to parse boundary information that matches the partitionIds we have
  static layer_comparator<0, tupletypeforbucketSize> pccomp2;
  auto leftBucketBoundaryRange = std::equal_range(allBoundaryPartitionSizes.begin(), allBoundaryPartitionSizes.end(), toSend[0], pccomp2);

  //Check if this processor owns this bucket
  if(std::get<1>(*(leftBucketBoundaryRange.first)) == rank)
  {
    iownLeftBucket = true;
    for(auto it = leftBucketBoundaryRange.first; it != leftBucketBoundaryRange.second; it++)
    {
      leftBucketSize += std::get<2>(*it);
    }
  }
  else
    iownLeftBucket = false;

  auto rightBucketBoundaryRange = std::equal_range(allBoundaryPartitionSizes.begin(), allBoundaryPartitionSizes.end(), toSend[1], pccomp2);

  //Check if this processor owns right partition
  if(std::get<1>(*rightBucketBoundaryRange.first) == rank && !onlySingleLocalPartition)
  {
    iownRightBucket = true;
    for(auto it = rightBucketBoundaryRange.first; it != rightBucketBoundaryRange.second; it++)
    {
      rightBucketSize += std::get<2>(*it);
    }
  }
  else
    iownRightBucket = false;

  //Start building the local histogram
  std::vector<uint64_t> localHistoGram;
  std::vector<uint64_t> histoGram;

  for(auto it = localVector.begin(); it!= localVector.end();)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, localVector.end(), *it, pccomp);

    //Left most bucket
    if (innerLoopBound.first == localVector.begin()) // first
    {
      if(iownLeftBucket)
        insertToHistogram(localHistoGram, leftBucketSize);
    }
    //Right most bucket
    else if (innerLoopBound.second == localVector.end()) // first
    {
      if(iownRightBucket)
        insertToHistogram(localHistoGram, rightBucketSize);
    }
    //Inner buckets
    else
    {
      insertToHistogram(localHistoGram, innerLoopBound.second - innerLoopBound.first);
    }

    it = innerLoopBound.second;
  }

  //Prepare for the final MPI_Reduce 
  uint32_t localSize= localHistoGram.size();
  uint32_t maxSize;

  //Find maximum size of local histogram
  MPI_Allreduce(&localSize, &maxSize, 1, MPI_UNSIGNED, MPI_MAX, comm); 
  localHistoGram.resize(maxSize);
  histoGram.resize(maxSize);

  //Do the reduction
  MPI_Reduce(&localHistoGram[0], &histoGram[0], maxSize, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm); 

  //Write to file
  if(rank == 0)
  {
    std::ofstream ofs;
    ofs.open(filename, std::ios_base::out);

    for(auto &e : histoGram)
    {
      ofs << e <<  "\n";
    }

    ofs.close();
  }
}



#endif
