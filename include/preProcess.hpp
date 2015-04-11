#ifndef PRE_PROCESS_HPP 
#define PRE_PROCESS_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp>

//Own includes
#include "sortTuples.hpp"


/*
 * @brief       Compute median coverage of kmers in a read. Trim off if greater than threshold.
 *              This process has been referred as digital normalization.
 *              This process also trims off read if there is any kmer with excess coverage.
 */
template <unsigned int kmerLayer, unsigned int tmpLayer, unsigned int readIdLayer, unsigned int medianCutOff = 10,  unsigned int frequencyCutOff = 50, typename T>
void trimReadswithHighMedianOrMaxCoverage(std::vector<T>& localvector, MPI_Comm comm = MPI_COMM_WORLD)
{
  /*
   * Approach:
   * 1. Sort by Kmers, count and save the frequency in tmpLayer
   * 2. Sort by readIdLayer, compute the median for each read
   * 3. If median is above threshold, place MAX in tmpLayer
   *    Else put readId in tmpLayer
   * 4. Partition and delete trimmed reads
   */

  int p;
  int rank;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &rank);

  static layer_comparator<kmerLayer, T> kmerCmp;
  static layer_comparator<readIdLayer, T> pcCmp;
  static layer_comparator<tmpLayer, T> tmpCmp;

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

  //Step 1 completed
  //Need to sort the data by readId and compute the medians

  mxx::sort(start, end, pcCmp, comm, false); 

  //Computing median over boundary is tricky
  //Lets not worry about boundary cases for now, p is << #Reads filtered out

  for(auto it = start; it!= end;)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, end, *it, pcCmp);

    //To compute the median of this bucket
    std::nth_element(innerLoopBound.first, innerLoopBound.first + std::distance(innerLoopBound.first, innerLoopBound.second)/2, innerLoopBound.second, tmpCmp);
    auto currentMedian = *(innerLoopBound.first + std::distance(innerLoopBound.first, innerLoopBound.second)/2);

    //Compute the max 
    auto currentMax = *std::max_element(innerLoopBound.first, innerLoopBound.second, tmpCmp);

    //Either mark the tuples as removed or restore their tmpLayer
    if(std::get<tmpLayer>(currentMedian) > medianCutOff || std::get<tmpLayer>(currentMax) > frequencyCutOff) 
      std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX;});
    else
      std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = std::get<readIdLayer>(t);});

    it = innerLoopBound.second;
  }

  //Partition the dataset and remove the abundant kmers
  BoundaryKmerPredicate<tmpLayer, T> correctReadPredicate;
  auto k_end = std::partition(start, end, correctReadPredicate);

  //Remove these kmers from vector
  localvector.erase(k_end, end);

  uint64_t localElementsRemoved = std::distance(k_end, end);

  auto totalKmersRemoved = mxx::reduce(localElementsRemoved);

  if(!rank) std::cerr << "[PREPROCESS: Digital Norm] " << totalKmersRemoved << " kmers removed from dataset\n"; 
}

/*
 * @brief             Remove kmers with frequency above threshold from the dataset
 * @details           Count the frequency of every kmer
 *                    Partition the ones which have higher frequence than the threshold
 *                    Then delete these from localVector
 * @tparam kmerLayer  kmer layer in the tuple
 * @tparam tmpLayer   Layer to mark high abundant kmer and remove eventually (Preferably Pn layer)
 * TODO               Digital normalization before this step
 */
template <unsigned int kmerLayer, unsigned int tmpLayer, unsigned int frequencyCutOff = 50, typename T>
void trimHighFrequencyKmer(std::vector<T>& localvector, MPI_Comm comm = MPI_COMM_WORLD)
{
  int p;
  int rank;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &rank);

  static layer_comparator<kmerLayer, T> kmerCmp;

  //Sort by kmer id
  mxx::sort(localvector.begin(), localvector.end(), kmerCmp, comm, false); 

  auto start = localvector.begin();
  auto end = localvector.end();
  auto rstart = localvector.rbegin();
  auto rend = localvector.rend();
  
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
      if(currentCount > frequencyCutOff)
      {
        std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX;});
      }
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

  if(leftBucketSize > frequencyCutOff)
  {
    //Update leftmost bucket
    auto innerLoopBound = findRange(start, end, *start, kmerCmp);
    std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX;});
  }

  //Right boundary case
  auto rightBucketBoundaryRange = std::equal_range(allBoundaryKmers.begin(), allBoundaryKmers.end(), toSend[1], gatherCmp);
 
  uint64_t rightBucketSize = 0;
  for(auto it = rightBucketBoundaryRange.first; it!= rightBucketBoundaryRange.second; it++) 
  {
    rightBucketSize += std::get<1>(*it);
  }

  if(rightBucketSize > frequencyCutOff)
  {
    //Update rightmost bucket
    auto innerLoopBound = findRange(rstart, rend, *rstart, kmerCmp);
    std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX;});
  }

  //Partition the dataset and remove the abundant kmers
  BoundaryKmerPredicate<tmpLayer, T> rightKmerPredicate;
  auto k_end = std::partition(start, end, rightKmerPredicate);

  //Remove these kmers from vector
  localvector.erase(k_end, end);

  uint64_t localElementsRemoved = std::distance(k_end, end);

  auto totalKmersRemoved = mxx::reduce(localElementsRemoved);

  if(!rank) std::cerr << "[PREPROCESS: TRIM HIGH FREQ. KMERS] " << totalKmersRemoved << " kmers removed from dataset\n"; 
}

#endif
