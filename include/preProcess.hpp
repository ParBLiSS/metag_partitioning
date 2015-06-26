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
 *                                In each bucket, the frequency is marekd as '1 2 ... totalCount' 
 *                                This is done to mimic progressive histogram equalization as done in khmer
 * @Note                          Don't change the local partition size during global sorts
 */
template <unsigned int kmerLayer, unsigned int tmpLayer, unsigned int readIdLayer, typename T>
void computeKmerFrequencyIncreasing(std::vector<T>& localvector, MPI_Comm comm = MPI_COMM_WORLD)
{
  //Know my rank
  int rank;
  MPI_Comm_rank(comm, &rank);

  static layer_comparator<kmerLayer, T> kmerCmp;

  auto start = localvector.begin();
  auto end = localvector.end();

  //Sort by kmer, read id
  mxx::sort(start, end, 
            [](const T& x, const T&y){
            return (std::get<kmerLayer>(x) < std::get<kmerLayer>(y)
                        || (std::get<kmerLayer>(x) == std::get<kmerLayer>(y)
                                    && std::get<readIdLayer>(x) < std::get<readIdLayer>(y)));
                    },
            comm, false); 


  //Keep frequency for leftmost and rightmost kmers seperately
  //Tuples of KmerId and count
  using tupleType = std::tuple<uint64_t, uint64_t>;
  std::vector<tupleType> toSend(2);
   
  for(auto it = start; it!= end;)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, end, *it, kmerCmp);

    //Leftmost bucket, appends information for sending to other ranks
    if(innerLoopBound.first == start)
    {
      std::get<0>(toSend[0]) = std::get<kmerLayer>(*start);
      std::get<1>(toSend[0]) = (innerLoopBound.second - innerLoopBound.first);
    }
    //Rightmost bucket
    else if(innerLoopBound.second == end)
    {
      std::get<0>(toSend[1]) = std::get<kmerLayer>(*innerLoopBound.first);
      std::get<1>(toSend[1]) = innerLoopBound.second - innerLoopBound.first;

      for(auto it1 = innerLoopBound.first; it1 != innerLoopBound.second; it1++)
        std::get<tmpLayer>(*it1) = std::distance(innerLoopBound.first, it1) + 1;
    }
    else
    {
      //Mark frequency as 1,2...totalCount
      for(auto it1 = innerLoopBound.first; it1 != innerLoopBound.second; it1++)
        std::get<tmpLayer>(*it1) = std::distance(innerLoopBound.first, it1) + 1;
    }

    it = innerLoopBound.second;
  }

  auto allBoundaryKmers = mxx::allgather_vectors(toSend);

  static layer_comparator<0, tupleType> gatherCmp;

  //Left boundary case
  //Count size only on the left side of this rank
  auto leftBucketBoundaryRange = std::equal_range(allBoundaryKmers.begin(), allBoundaryKmers.begin() + 2*rank, toSend[0], gatherCmp);

  uint64_t leftBucketSize = 0;
  for(auto it = leftBucketBoundaryRange.first; it != leftBucketBoundaryRange.second; it++)
  {
    leftBucketSize += std::get<1>(*it);
  }

  {
    //Update leftmost bucket
    auto innerLoopBound = findRange(start, end, *start, kmerCmp);
    for(auto it = innerLoopBound.first; it != innerLoopBound.second; it ++)
      std::get<tmpLayer>(*it) = leftBucketSize + (std::distance(innerLoopBound.first, it) + 1);
  }
}

/*
 * @brief                         Computes and saves the frequency of each kmer in the tmpLayer
 *                                Does a sort by kmerLayer to do the counting
 *                                In each bucket, the frequency is marekd as 'totalCount totalCount ... totalCount' 
 */
template <unsigned int kmerLayer, unsigned int tmpLayer, typename T>
void computeKmerFrequencyAbsolute(std::vector<T>& localvector, MPI_Comm comm = MPI_COMM_WORLD)
{
  //Know my rank
  int rank;
  MPI_Comm_rank(comm, &rank);

  static layer_comparator<kmerLayer, T> kmerCmp;

  auto start = localvector.begin();
  auto end = localvector.end();

  //Sort by kmer id
  mxx::sort(start, end, kmerCmp, comm, false); 


  //Keep frequency for leftmost and rightmost kmers seperately
  //Tuples of KmerId and count
  using tupleType = std::tuple<uint64_t, uint64_t>;
  std::vector<tupleType> toSend(2);
   
  for(auto it = start; it!= end;)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, end, *it, kmerCmp);

    //Leftmost bucket, appends information for sending to other ranks
    if(innerLoopBound.first == start)
    {
      std::get<0>(toSend[0]) = std::get<kmerLayer>(*start);
      std::get<1>(toSend[0]) = (innerLoopBound.second - innerLoopBound.first);
    }
    //Rightmost bucket
    else if(innerLoopBound.second == end)
    {
      std::get<0>(toSend[1]) = std::get<kmerLayer>(*innerLoopBound.first);
      std::get<1>(toSend[1]) = innerLoopBound.second - innerLoopBound.first;

      uint64_t currentCount = innerLoopBound.second - innerLoopBound.first; 
      std::for_each(innerLoopBound.first, innerLoopBound.second, [currentCount](T &t){ std::get<tmpLayer>(t) = currentCount;});
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
  //Count size only on the left side of this rank
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
}

/*
 * @brief                         Assumes kmer frequency in the tmpLayer, filters the reads based on the abundance of their kmers
 * @tparam[in] filterbyMedian     Filter if median exceeds "HIST_EQ_THRESHOLD" in the Pc bucket
 * @tparam[in] filterbyMax        Filter if maximum kmer frequence exceeds "KMER_FREQ_THRESHOLD" in the Pc bucket (which is a read during preprocessing)
 */
template <unsigned int tmpLayer, unsigned int readIdLayer, unsigned int kmerSnoLayer, bool filterbyMedian, bool filterbyMax, typename T>
void updateReadFilterFlags(std::vector<T>& localvector, std::vector<bool>& readFilterFlags, std::vector<ReadLenType>& readTrimLengths,
                          uint32_t firstReadId, 
                          MPI_Comm comm = MPI_COMM_WORLD)
{
  //Know my rank
  int rank;
  MPI_Comm_rank(comm, &rank);

  //Custom comparators for tuples inside localvector
  static layer_comparator<readIdLayer, T> readCmp;
  static layer_comparator<tmpLayer, T> freqCmp;

  auto start = localvector.begin();
  auto end = localvector.end();

  //Need to sort the data by readId,kmer-Sno
  mxx::sort(start, end, 
      [](const T& x, const T&y){
      return (std::get<readIdLayer>(x) < std::get<readIdLayer>(y)
        || (std::get<readIdLayer>(x) == std::get<readIdLayer>(y)
          && std::get<kmerSnoLayer>(x) < std::get<kmerSnoLayer>(y)));}
      , comm, false); 


  //Since the partitioning size haven't been modified yet, the partitions shouldn't spawn 
  //across processors

  //Track count of kmers removed
  uint64_t localElementsRemoved = 0;

  for(auto it = start; it!= end;)  // iterate over all segments.
  {
    auto innerLoopBound = findRange(it, end, *it, readCmp);

    //To compute the median of this bucket
    if (filterbyMedian) std::nth_element(innerLoopBound.first, innerLoopBound.first + std::distance(innerLoopBound.first, innerLoopBound.second)/2, innerLoopBound.second, freqCmp);
    auto currentMedian = *(innerLoopBound.first + std::distance(innerLoopBound.first, innerLoopBound.second)/2);

    //Compute the max
    auto currentMax = std::max_element(innerLoopBound.first, innerLoopBound.second, freqCmp);

    //Read id
    auto readId = std::get<readIdLayer>(*innerLoopBound.first);

    //Either mark the tuples as removed or restore their tmpLayer
    if(filterbyMedian && std::get<tmpLayer>(currentMedian) > HIST_EQ_THRESHOLD) 
    {
      readFilterFlags[readId - firstReadId] = false;
      localElementsRemoved += 1;
      readTrimLengths[readId - firstReadId] = 0; 

      //Mark the flag inside tuple to delete later
      std::for_each(innerLoopBound.first, innerLoopBound.second, [](T &t){ std::get<tmpLayer>(t) = MAX_FREQ;});
    }
    else if(filterbyMax &&  std::get<tmpLayer>(*currentMax) > KMER_FREQ_THRESHOLD)
    {
      readFilterFlags[readId - firstReadId] = false;
      localElementsRemoved += 1;

      //Find first element greater than KMER_FREQ_THRESHOLD
      auto firstHighFreq= std::find_if(innerLoopBound.first, innerLoopBound.second, [](T &t){return std::get<tmpLayer>(t) > KMER_FREQ_THRESHOLD;});

      readTrimLengths[readId - firstReadId] = KMER_LEN_PRE + (firstHighFreq - innerLoopBound.first) - 1; 
    }
    else
    {
      readFilterFlags[readId - firstReadId] = true;
    }

    it = innerLoopBound.second;
  }

  //Count the reads filtered
  auto totalReadsRemoved = mxx::reduce(localElementsRemoved);

  if(!rank) std::cerr << "[PREPROCESS: ] " << totalReadsRemoved << " reads removed/trimmed from dataset\n"; 

}

/// partition predicate for trimming/removing reads.
template<uint8_t activePartitionLayer, typename T>
struct KeepKmerPredicate{
  bool operator()(const T& x) {
    return std::get<activePartitionLayer>(x) < MAX_FREQ;
  }
};

/*
 * @brief                         Compute median coverage of kmers in a read. Discard the read if it is greater than "medianCutOff".
 *                                This process has been referred as digital normalization.
 *                                After completing the above step, this function also trims off read 
 *                                if there is any kmer with coverage more than "frequencyCutOff"
 * @param[out] readFilterFlags    Recall every process parses reads. If a rank parses say 10 reads, 
 *                                readFilterFlags is bool vector of size 10, and is marked as false if read should be discarded
 */
template <typename T, unsigned int kmerLayer = kmerTuple_Pre::kmer, unsigned int tmpLayer = kmerTuple_Pre::freq, unsigned int readIdLayer = kmerTuple_Pre::rid, unsigned int kmerSnoLayer = kmerTuple_Pre::kmer_sno>
void trimReadswithHighMedianOrMaxCoverage(std::vector<T>& localvector, std::vector<bool>& readFilterFlags, std::vector<ReadLenType>& readTrimLengths, MPI_Comm comm = MPI_COMM_WORLD)
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
  readTrimLengths.resize(lastReadId - firstReadId + 1);

  //Compute kmer frequency
  MP_TIMER_START();
  computeKmerFrequencyIncreasing<kmerLayer, tmpLayer, readIdLayer>(localvector); 
  MP_TIMER_END_SECTION("[PREPROCESS TIMER] First kmer counting pass completed");

  //Perform the digital normalization
  updateReadFilterFlags<tmpLayer, readIdLayer, kmerSnoLayer, true, false>(localvector, readFilterFlags, readTrimLengths, firstReadId);
  MP_TIMER_END_SECTION("[PREPROCESS TIMER] Digi-Norm completed");

  //Remove the marked reads from the working dataset
  KeepKmerPredicate<tmpLayer, T> readFilterPredicate;
  auto k_end = std::partition(localvector.begin(), localvector.end(), readFilterPredicate);
  localvector.erase(k_end, localvector.end());

  //Re-compute kmer frequency after trimming reads during digital normalization
  //No need for a barrier at this point
  computeKmerFrequencyAbsolute<kmerLayer, tmpLayer>(localvector); 
  MP_TIMER_END_SECTION("[PREPROCESS TIMER] Second kmer counting pass completed");

  //Filter out only low abundant kmers
  updateReadFilterFlags<tmpLayer, readIdLayer, kmerSnoLayer, false, true>(localvector, readFilterFlags, readTrimLengths, firstReadId);
  MP_TIMER_END_SECTION("[PREPROCESS TIMER] Filtering low abundant reads completed");
}

#endif
