#ifndef PRE_PROCESS_HPP 
#define PRE_PROCESS_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp>

//Own includes
#include "sortTuples.hpp"

/*
 * @brief                         Computes and saves the frequency of each kmer in the tmpLayer
 *                                Does a sort by kmerLayer to do the counting
 * @Note                          Don't change the local partition size during global sorts
 */
template <unsigned int kmerLayer, unsigned int tmpLayer, typename T>
void computeKmerFrequency(std::vector<T>& localvector, MPI_Comm comm = MPI_COMM_WORLD)
{
  static layer_comparator<kmerLayer, T> kmerCmp;

  auto start = localvector.begin();
  auto end = localvector.end();
  auto rstart = localvector.rbegin();
  auto rend = localvector.rend();

  //Sort by kmer id
  mxx::sort(start, end, kmerCmp, comm, false); 


  //Keep frequency for leftmost and rightmost kmers seperately
  //Tuples of KmerId and count
  using tupleType = std::tuple<uint64_t, uint64_t>;
  std::vector<tupleType> toSend(2);
   
  for(auto it = start; it!= end;)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, end, *it, kmerCmp);

    //Leftmost bucket
    if(innerLoopBound.first == start)
    {
      std::get<0>(toSend[0]) = std::get<kmerLayer>(*start);
      std::get<1>(toSend[0]) = (innerLoopBound.second - innerLoopBound.first);
    }
    //Rightmost bucket
    else if(innerLoopBound.second == end)
    {
      std::get<0>(toSend[1]) = std::get<kmerLayer>(*innerLoopBound.first);

      //Corner case of having same kmer through out
      if(std::get<kmerLayer>(*start) !=  std::get<kmerLayer>(*innerLoopBound.first))
        std::get<1>(toSend[1]) = innerLoopBound.second - innerLoopBound.first;
    }
    else
    {
      uint64_t currentCount = innerLoopBound.second - innerLoopBound.first; 
      std::for_each(innerLoopBound.first, innerLoopBound.second, [currentCount](T &t){ std::get<tmpLayer>(t) = currentCount;});
    }

    it = innerLoopBound.second;
  }

  auto allBoundaryKmers = mxx::allgather_vectors(toSend);

  static layer_comparator<0, tupleType> gatherCmp;

  //Left boundary case
  auto leftBucketBoundaryRange = std::equal_range(allBoundaryKmers.begin(), allBoundaryKmers.end(), toSend[0], gatherCmp);

  uint64_t leftBucketSize = 0;
  for(auto it = leftBucketBoundaryRange.first; it != leftBucketBoundaryRange.second; it++)
  {
    leftBucketSize += std::get<1>(*it);
  }

  {
    //Update leftmost bucket
    auto innerLoopBound = findRange(start, end, *start, kmerCmp);
    std::for_each(innerLoopBound.first, innerLoopBound.second, [leftBucketSize](T &t){ std::get<tmpLayer>(t) = leftBucketSize;});
  }

  //Right boundary case
  auto rightBucketBoundaryRange = std::equal_range(allBoundaryKmers.begin(), allBoundaryKmers.end(), toSend[1], gatherCmp);
 
  uint64_t rightBucketSize = 0;
  for(auto it = rightBucketBoundaryRange.first; it!= rightBucketBoundaryRange.second; it++) 
  {
    rightBucketSize += std::get<1>(*it);
  }

  {
    //Update rightmost bucket
    auto innerLoopBound = findRange(rstart, rend, *rstart, kmerCmp);
    std::for_each(innerLoopBound.first, innerLoopBound.second, [rightBucketSize](T &t){ std::get<tmpLayer>(t) = rightBucketSize;});
  }
}

/*
 * @brief                         Assumes kmer frequency in the tmpLayer, filters the reads based on the abundance of their kmers
 * @tparam[in] filterbyMedian     Filter if median exceeds "medianCutOff" in the Pc bucket
 * @tparam[in] filterbyMax        Filter if maximum kmer frequence exceeds "frequencyCutOff" in the Pc bucket (which is a read during preprocessing)
 */
template <unsigned int tmpLayer, unsigned int readIdLayer, bool filterbyMedian, bool filterbyMax, uint32_t medianCutOff = HIST_EQ_THRESHOLD, uint32_t frequencyCutOff = KMER_FREQ_THRESHOLD, typename T>
void updateReadFilterFlags(std::vector<T>& localvector, std::vector<bool>& readFilterFlags, uint32_t firstReadId, MPI_Comm comm = MPI_COMM_WORLD)
{
  //Know my rank
  int rank;
  MPI_Comm_rank(comm, &rank);

  //Custom comparators for tuples inside localvector
  static layer_comparator<readIdLayer, T> pcCmp;
  static layer_comparator<tmpLayer, T> tmpCmp;

  auto start = localvector.begin();
  auto end = localvector.end();

  //Need to sort the data by readId and compute the medians
  mxx::sort(start, end, pcCmp, comm, false); 


  //Since the partitioning size haven't been modified yet, the partitions shouldn't spawn 
  //across processors

  //Track count of kmers removed
  uint64_t localElementsRemoved = 0;

  for(auto it = start; it!= end;)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, end, *it, pcCmp);

    //To compute the median of this bucket
    std::nth_element(innerLoopBound.first, innerLoopBound.first + std::distance(innerLoopBound.first, innerLoopBound.second)/2, innerLoopBound.second, tmpCmp);
    auto currentMedian = *(innerLoopBound.first + std::distance(innerLoopBound.first, innerLoopBound.second)/2);

    //Compute the max
    auto currentMax = *std::max_element(innerLoopBound.first, innerLoopBound.second, tmpCmp);

    std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = std::get<readIdLayer>(t);});

    //Read id
    auto readId = std::get<readIdLayer>(*innerLoopBound.first);

    //Either mark the tuples as removed or restore their tmpLayer
    if(filterbyMedian && std::get<tmpLayer>(currentMedian) > medianCutOff) 
    {
      readFilterFlags[readId - firstReadId] = false;
      localElementsRemoved += 1;

      //Mark the flag inside tuple to delete later
      std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX;});
    }
    else if(filterbyMax &&  std::get<tmpLayer>(currentMax) > frequencyCutOff)
    {
      readFilterFlags[readId - firstReadId] = false;
      localElementsRemoved += 1;

      //Mark the flag inside tuple to delete later
      std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX;});
    }
    else
    {
      readFilterFlags[readId - firstReadId] = true;

      //Restore tmpLayer
      std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = std::get<readIdLayer>(t);});
    }

    it = innerLoopBound.second;
  }

  //Count the reads filtered
  auto totalReadsRemoved = mxx::reduce(localElementsRemoved);

  if(!rank) std::cerr << "[PREPROCESS: ] " << totalReadsRemoved << " reads removed from dataset\n"; 

}

/*
 * @brief                         Compute median coverage of kmers in a read. Discard the read if it is greater than "medianCutOff".
 *                                This process has been referred as digital normalization.
 *                                After completing the above step, this function also trims off read 
 *                                if there is any kmer with coverage more than "frequencyCutOff"
 * @param[out] readFilterFlags    Recall every process parses reads. If a rank parses say 10 reads, 
 *                                readFilterFlags is bool vector of size 10, and is marked as false if read should be discarded
 */
template <typename T, unsigned int kmerLayer = 0, unsigned int tmpLayer = 1, unsigned int readIdLayer = 2>
void trimReadswithHighMedianOrMaxCoverage(std::vector<T>& localvector, std::vector<bool>& readFilterFlags, MPI_Comm comm = MPI_COMM_WORLD)
{
  /*
   * Approach:
   * 1. Sort by Kmers, count and save the frequency in tmpLayer
   * 2. Sort by readIdLayer, compute the median for each read
   * 3. If median is above threshold, place MAX in tmpLayer
   *    Else put readId in tmpLayer
   * 4. Partition and delete trimmed reads
   */

  //How many reads this processor has parsed while building the index
  auto lastReadId = std::get<readIdLayer>(localvector.back());
  auto firstReadId = std::get<readIdLayer>(localvector.front());

  //Preserve the size of readFilterFlags based on the above count
  readFilterFlags.resize(lastReadId - firstReadId + 1);

  //Compute kmer frequency
  computeKmerFrequency<kmerLayer, tmpLayer>(localvector); 

  //Perform the digital normalization
  updateReadFilterFlags<tmpLayer, readIdLayer, true, false>(localvector, readFilterFlags, firstReadId);

  //Remove the marked reads from the working dataset
  BoundaryKmerPredicate<tmpLayer, T> readFilterPredicate;
  auto k_end = std::partition(localvector.begin(), localvector.end(), readFilterPredicate);
  localvector.erase(k_end, localvector.end());

  //Re-compute kmer frequency after trimming reads during digital normalization
  computeKmerFrequency<kmerLayer, tmpLayer>(localvector); 

  //Filter out only low abundant kmers
  updateReadFilterFlags<tmpLayer, readIdLayer, false, true>(localvector, readFilterFlags, firstReadId);
}

#endif
