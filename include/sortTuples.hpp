#ifndef SORT_TUPLES_HPP 
#define SORT_TUPLES_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp> 
#include <mxx/shift.hpp> 

//Own includes
#include "prettyprint.hpp"

#include <fstream>
#include <iostream>

static char dummyBool;



/**
 * @details
 * localVector is a vector of tuples (R, Kmer, Pn, Pc) on each MPI process.
 * Notation : Layer i would denote all ith elements of the tuples inside vector
 * Goal of this function is :
 *  to sort all the kmers by "sortLayer"th layer
 *  Within the buckets, look for minimum element in "pickMinLayer"th layer
 *  Make all "pickMinLayer"th element equal to minimum element picked
 *
 *  If the Sortlayer is the <..,..,..,Pc>, updateSortLayer should be set to true
 *  if updateSortLayer is true, also make "SortLayer"th elements equal to minimum
 *
 * @rational
 * This function can be used to perform all the sorts needed for partitioning
 */
template <uint8_t sortLayer, uint8_t pickMinLayer, bool updateSortLayer=false, typename T>
void sortTuples(std::vector<T>& localVector, char& wasSortLayerUpdated = dummyBool, MPI_Comm comm = MPI_COMM_WORLD)
{
  //Define a less comparator for ordering tuples based on sortLayer
  auto comparator = [](const T& x, const T& y){
    return std::get<sortLayer>(x) < std::get<sortLayer>(y);
  };

  //Sort the vector by each tuple's "sortLayer"th element
  mxx::sort(localVector.begin(), localVector.end(), comparator, comm, false);

  /// MPI rank within the communicator
  int rank, commsize;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &commsize);

  // Save left and right bucket Ids for later boundary value handling
  auto leftMostBucketId = std::get<sortLayer>(localVector.front());
  auto rightMostBucketId = std::get<sortLayer>(localVector.back());

  //Initialise loop variables

  //Find the minimum element on pickMinLayer in each bucket
  //Update elements in each bucket to the minima

  //Useful for checking termination of the algorithm
  if(updateSortLayer)
    wasSortLayerUpdated = 0;

  for(auto it = localVector.begin(); it!= localVector.end();)
  {
    //See the first element and find its bucket members ahead
    //Report the range of the bucket
    //Shouldn't lead to any extra overhead because range starts from the beginning of iterator
    auto innerLoopBound = std::equal_range(it, localVector.end(), *it, comparator);

    //Scan this bucket and find the minimum
    auto currentMinimum = std::get<pickMinLayer>(*it);
    for(auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; ++it2)
      currentMinimum = std::min(currentMinimum, std::get<pickMinLayer>(*it2));

    //Again scan this bucket to assign the minimum found above
    for(auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; ++it2)
    {
      //Update the pickMinLayer elements
      std::get<pickMinLayer>(*it2) = currentMinimum;

      //If sortLayer also needs to be updated
      if(updateSortLayer)
      {
        if(std::get<sortLayer>(*it2) != currentMinimum)
        {
          //Set the boolean flag and update the value
          wasSortLayerUpdated = 1;
          std::get<sortLayer>(*it2) = currentMinimum;
        }
      }
    }

    //Forward the outer iterator to beginning of next bucket
    it = innerLoopBound.second;
  }


  MPI_Barrier(comm);

  /***
   * NOW WE DEAL WITH STUPID BOUNDARY VALUES  - TODO: use ALL REDUCE
   **/

  //Since the buckets could be partitioned across processors, we need to do some more work
  //to ensure correctness

  //Approach: Let each adjacent processors exchange their boundary values
  //Based on the value received, they can decide and update their last (and first) buckets respectively

  auto tupleFromLeft = mxx::right_shift(localVector.back());
  auto tupleFromRight = mxx::left_shift(localVector.front());

  //Get the bucketIds as well to match them
  auto bucketIdfromLeft = mxx::right_shift(rightMostBucketId);
  auto bucketIdfromRight = mxx::left_shift(leftMostBucketId);

  //Deal with tuple coming from left side first
  if(rank > 0)
  {
    //Check if tuple received belongs to same bucket as my first element
    bool cond1 = bucketIdfromLeft == leftMostBucketId;

    //Check if tuple received has smaller value, only then proceed
    bool cond2 = std::get<pickMinLayer>(tupleFromLeft) < std::get<pickMinLayer>(localVector.front());

    if(cond1 && cond2)
    {
      //The code to update the bucket is same as before
      auto innerLoopBound = std::equal_range(localVector.begin(), localVector.end(), *(localVector.begin()), comparator);
      for(auto it = innerLoopBound.first; it != innerLoopBound.second; ++it)
      {
        std::get<pickMinLayer>(*it) = std::get<pickMinLayer>(tupleFromLeft);
        if(updateSortLayer)
        {
          if(std::get<sortLayer>(*it) != std::get<pickMinLayer>(tupleFromLeft))
          {
            //Set the boolean flag and update the value
            wasSortLayerUpdated = 1;
            std::get<sortLayer>(*it) = std::get<pickMinLayer>(tupleFromLeft);
          }
        }
      }
    }
  }

  //Deal with tuple coming from right side now
  if(rank < commsize - 1)
  {
    //Check if tuple received belongs to same bucket as my first element
    bool cond1 = bucketIdfromRight == rightMostBucketId;

    //Check if tuple received has smaller value, only then proceed
    bool cond2 = std::get<pickMinLayer>(tupleFromRight) < std::get<pickMinLayer>(localVector.back());

    if(cond1 && cond2)
    {
      //The code to update the bucket is same as before
      auto innerLoopBound = std::equal_range(localVector.rbegin(), localVector.rend(), *(localVector.rbegin()), comparator);
      for(auto it = innerLoopBound.first; it != innerLoopBound.second; ++it)
      {
        std::get<pickMinLayer>(*it) = std::get<pickMinLayer>(tupleFromRight);
        if(updateSortLayer)
        {
          if(std::get<sortLayer>(*it) != std::get<pickMinLayer>(tupleFromRight))
          {
            //Set the boolean flag and update the value
            wasSortLayerUpdated = 1;
            std::get<sortLayer>(*it) = std::get<pickMinLayer>(tupleFromRight);
          }
        }
      }
    }
  }
}

template<uint8_t layer, typename T >
struct layer_comparator : public std::binary_function<T, T, bool>
{
    bool operator()(const T& x, const T& y) {
      return std::get<layer>(x) < std::get<layer>(y);
    }
};

/**
 * Reducer functor for merging partitions that shared the same kmer.
 * Pc = reductionLayer, Pn = resultLayer, flagLayer indicate whether the kmer / partition is active or not.
 *
 * do a local reduction for the entire range with multiple "segments".
 * next do an all gather to get the boundaries
 * then do a local reduction for each global segment
 * and update local segments and update the reduction.
 *
 *
 */
template <uint8_t keyLayer, uint8_t reductionLayer, uint8_t resultLayer, uint8_t activeLayer, typename T>
struct KmerReduceAndMarkAsInactive {


    void operator()(typename std::vector<T>::iterator start, typename std::vector<T>::iterator end,
                    MPI_Comm comm = MPI_COMM_WORLD) {

      int p;
      int rank;
      MPI_Comm_size(comm, &p);
      MPI_Comm_rank(comm, &rank);

      layer_comparator<keyLayer, T> keycomp;
      layer_comparator<reductionLayer, T>  pccomp;
      layer_comparator<resultLayer, T> pncomp;

      // init storage
      std::vector<T> toSend(2);
      std::vector<T> toRecv(2 * p);

      T v = *start;
      auto minPc = std::get<reductionLayer>(v);
      auto maxPc = minPc;
      auto y = std::get<activeLayer>(v);  // bit 0 for kmer internal/boundary (0), bit 1 for partition inactive/active (0)
      // so 2,3 indicate partition is inactive,
      // 0 is active partition, boundary kmer
      // 1 is active partition, internal kmer


      auto lastStart = start;
      auto firstEnd = end;

      auto innerLoopBound = std::equal_range(start, end, *start, keycomp);


      for(auto it = start; it!= end;)  // iterate over all segments.
      {
        y = std::get<activeLayer>(*it);
        // if 1 of the tuples have been marked as internal, all tuples with the same kmer are marked as internal
        assert(y < 0x2);  //  inactive partition, should NOT get here.

        // else a kmer at partition boundary.

        // get the range with the first element's key value.
        innerLoopBound = std::equal_range(it, end, *it, keycomp);

        // Scan this bucket and find the minimum and max.  min vs max to see if kmer is internal
        auto minmax = std::minmax_element(innerLoopBound.first, innerLoopBound.second, pccomp);
        minPc = std::get<reductionLayer>(*(minmax.first));
        maxPc = std::get<reductionLayer>(*(minmax.second));

        // if minPc == maxPc, then all Pc are equal, and we have an internal kmer (in active partition).  mark it.
        // else it's a boundary kmer.
        y = (minPc == maxPc) ? 0x1 : 0x0;

        if (innerLoopBound.first != start && innerLoopBound.second != end)  // middle
        {
          // can update directly.

          // then update all entries in bucket

          // update all kmers Pn in this range.
          for (auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; ++it2) {
            std::get<activeLayer>(*it2) = y;
            std::get<resultLayer>(*it2) = minPc;
          }
        }

        if (innerLoopBound.first == start) // first
        {
          // save the first entry
          firstEnd = innerLoopBound.second;
          std::get<keyLayer>(toSend[0]) = std::get<keyLayer>(*(innerLoopBound.first));
          std::get<reductionLayer>(toSend[0]) = maxPc;
          std::get<resultLayer>(toSend[0]) = minPc;
        }

        if (innerLoopBound.second == end)  // last
        {
          // save the last entry (actually, first entry in the bucket.
          lastStart = innerLoopBound.first;
          std::get<keyLayer>(toSend[1]) = std::get<keyLayer>(*(innerLoopBound.first));
          std::get<reductionLayer>(toSend[1]) = maxPc;
          std::get<resultLayer>(toSend[1]) = minPc;
        }

        it =  innerLoopBound.second;

      }


      // global gather of values from all the mpi processes, just the first and second.
      // get MPI type
      mxx::datatype<T> dt;
      MPI_Datatype mpi_dt = dt.type();
      MPI_Allgather(&(toSend[0]), 2, mpi_dt, &(toRecv[0]), 2 * p, mpi_dt, comm);


      // local reduction of global data.  only need to do the range that this mpi process is related to.
      // array should already be sorted by k since this is a sampling at process boundaries of a distributed sorted array.


      // first group in local.
        innerLoopBound = std::equal_range(toRecv.begin(), toRecv.end(), toSend[0], keycomp);

        minPc = std::get<resultLayer>(*(std::min_element(innerLoopBound.first, innerLoopBound.second, pncomp)));
        maxPc = std::get<reductionLayer>(*(std::max_element(innerLoopBound.first, innerLoopBound.second, pccomp)));

        // update the first group with maxPC and minPC.
        // if minPc == maxPc, then all Pc are equal, and we have an internal kmer (in active partition).  mark it.
        // else it's a boundary kmer.
        y = (minPc == maxPc) ? 0x1 : 0x0;
        for(auto it2 = start; it2 != firstEnd; ++it2)
        {
          std::get<resultLayer>(*it2) = minPc;
          std::get<activeLayer>(*it2) = y;
        }


        // last group in localvector
        if (std::get<keyLayer>(toSend[0]) != std::get<keyLayer>(toSend[1])) {

          // get each bucket
          innerLoopBound = std::equal_range(innerLoopBound.second, toRecv.end(), toSend[0], keycomp);

          minPc = std::get<resultLayer>(*(std::min_element(innerLoopBound.first, innerLoopBound.second, pncomp)));
          maxPc = std::get<reductionLayer>(*(std::max_element(innerLoopBound.first, innerLoopBound.second, pccomp)));

          // if minPc == maxPc, then all Pc are equal, and we have an internal kmer (in active partition).  mark it.
          // else it's a boundary kmer.
          y = (minPc == maxPc) ? 0x1 : 0x0;
          for(auto it2 = lastStart; it2 != end; ++it2)
          {
            std::get<resultLayer>(*it2) = minPc;
            std::get<activeLayer>(*it2) = y;
          }
        }

    }
};


/**
 * do a local reduction for the entire range with multiple "segments".
 * next do an all gather to get the boundaries
 * then do a local reduction for each global segment
 * and update local segments and update the reduction.
 *
 *
 */
template <uint8_t keyLayer, uint8_t reductionLayer, uint8_t activeLayer, typename T>
struct PartitionReduceAndMarkAsInactive {


    void operator()(typename std::vector<T>::iterator start, typename std::vector<T>::iterator end,
                    MPI_Comm comm = MPI_COMM_WORLD) {

      int p;
      int rank;
      MPI_Comm_size(comm, &p);
      MPI_Comm_rank(comm, &rank);

      layer_comparator<keyLayer, T> keycomp;
      layer_comparator<reductionLayer, T>  pncomp;
      layer_comparator<activeLayer, T> flagcomp;

      // init storage
      std::vector<T> toSend(2);
      std::vector<T> toRecv(2 * p);

      T v = *start;
      auto minPn = std::get<reductionLayer>(v);  // Pn
      auto y = std::get<activeLayer>(v);  // bit 0 for kmer internal/boundary (0), bit 1 for partition inactive/active (0)
      // so 2,3 indicate partition is inactive,
      // 0 is active partition, boundary kmer
      // 1 is active partition, internal kmer
      auto minY = y;

      auto lastStart = start;
      auto firstEnd = end;

      auto innerLoopBound = std::equal_range(start, end, *start, keycomp);  // search by Pc

      // do reduction on all partition segments
      for(auto it = start; it!= end;)
      {

        y = std::get<activeLayer>(*it);

        // if 1 of the tuples have been marked as inactive, the entire partition must be marked as inactive.
        // should not get here.
        assert(y < 0x2);


        // else active partition to update it.

        // get the range with the first element's key value.
        innerLoopBound = std::equal_range(it, end, *it, keycomp); // get segment for Pc

        // Scan this bucket and find the minimum
        minPn = std::get<reductionLayer>(*(std::min_element(innerLoopBound.first, innerLoopBound.second, pncomp)));  // get min Pn
        minY = std::get<activeLayer>(*(std::min_element(innerLoopBound.first, innerLoopBound.second, flagcomp)));  // get min Pn


        if (innerLoopBound.first != start && innerLoopBound.second != end)  // middle
        {
          // can update directly.

          // then update all entries in bucket
          for (auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; ++it2) {
            // if partition has only internal kmers, then this partition is to become inactive..
            if (minY > 0x0) std::get<activeLayer>(*it2) |= 0x2;
            std::get<keyLayer>(*it2) = minPn;  // new partition id.
          }
        }

        if (innerLoopBound.first == start) // first
        {
          // save the first entry
          firstEnd = innerLoopBound.second;
          std::get<keyLayer>(toSend[0]) = std::get<keyLayer>(*(innerLoopBound.first));
          std::get<reductionLayer>(toSend[0]) = minPn;
          std::get<activeLayer>(toSend[0]) = minY;
        }

        if (innerLoopBound.second == end)  // last
        {
          // save the last entry (actually, first entry in the bucket.
          lastStart = innerLoopBound.first;
          std::get<keyLayer>(toSend[1]) = std::get<keyLayer>(*(innerLoopBound.first));
          std::get<reductionLayer>(toSend[1]) = minPn;
          std::get<activeLayer>(toSend[1]) = minY;
        }

        it =  innerLoopBound.second;

      }


      // global gather of values from all the mpi processes, just the first and second.
      // get MPI type
      mxx::datatype<T> dt;
      MPI_Datatype mpi_dt = dt.type();
      MPI_Allgather(&(toSend[0]), 2, mpi_dt, &(toRecv[0]), 2 * p, mpi_dt, comm);


      // local reduction of global data.  only need to do the range that this mpi process is related to.
      // array should already be sorted by k since this is a sampling at process boundaries of a distributed sorted array.


      // first group in local.
      innerLoopBound = std::equal_range(toRecv.begin(), toRecv.end(), toSend[0], keycomp);

      minY = std::get<activeLayer>(*(std::min_element(innerLoopBound.first, innerLoopBound.second, flagcomp)));
      minPn = std::get<reductionLayer>(*(std::max_element(innerLoopBound.first, innerLoopBound.second, pncomp)));

      // if all kmers in partition are internal, then the partition is marked inactive.
      for(auto it2 = start; it2 != firstEnd; ++it2)
      {
        // if minY shows inactive partition or internal kmers, then this partition is to become inactive..
        if (minY > 0x0) std::get<activeLayer>(*it2) |= 0x2;
        std::get<keyLayer>(*it2) = minPn;  // new partition id.
      }

      // last group in localvector
      if (std::get<keyLayer>(toSend[0]) != std::get<keyLayer>(toSend[1])) {
        innerLoopBound = std::equal_range(innerLoopBound.second, toRecv.end(), toSend[1], keycomp);

        minY = std::get<activeLayer>(*(std::min_element(innerLoopBound.first, innerLoopBound.second, flagcomp)));
        minPn = std::get<reductionLayer>(*(std::max_element(innerLoopBound.first, innerLoopBound.second, pncomp)));

        // if all kmers in partition are internal, then the partition is marked inactive.
        for(auto it2 = lastStart; it2 != end; ++it2)
        {
          // if minY shows inactive partition or internal kmers, then this partition is to become inactive..
          if (minY > 0x0) std::get<activeLayer>(*it2) |= 0x2;
          std::get<keyLayer>(*it2) = minPn;  // new partition id.
        }

      }

    }
};




/// partition predicate for active Pc.  all inactive partitions are move to the front of iterator.
template<uint8_t activePartitionLayer, typename T>
struct ActivePartitionPredicate {
    bool operator()(const T& x) {
      auto v = std::get<activePartitionLayer>(x);
      return (v & 0x2) > 0;
    }
};





/**
 * @details
 * start and end are iterators for a vector of tuples (R, Kmer, Pn, Pc) on each MPI process.
 * Notation : Layer i would denote all ith elements of the tuples inside vector
 * Goal of this function is :
 *  to sort all the kmers by "sortLayer"th layer
 *  Within the buckets, look for minimum element in "pickMinLayer"th layer
 *  Make all "pickMinLayer"th element equal to minimum element picked if changes,
 *      otherwise mark it as max-1 (inactive K, active Pc)
 *
 *  If the Sortlayer is the <..,..,..,Pc>, updateSortLayer should be set to true
 *  if updateSortLayer is true, also make "SortLayer"th elements equal to minimum
 *
 * @rational
 * This function can be used to perform all the sorts needed for partitioning
 */
template <uint8_t sortLayer, typename Reducer,  typename T>
void sortAndReduceTuples(typename std::vector<T>::iterator start, typename std::vector<T>::iterator end,
                         MPI_Comm comm = MPI_COMM_WORLD)
{

  //Sort the vector by each tuple's "sortLayer"th element
  mxx::sort(start, end, layer_comparator<sortLayer, T>(), comm, false);

  // local reduction
  //Find the minimum element on pickMinLayer in each bucket
  //Update elements in each bucket to the minima
  Reducer r;
  r(start, end, comm);

}

/// do global reduction to see if termination criteria is satisfied.
template<uint8_t terminationFlagLayer, typename T>
bool checkTermination(typename std::vector<T>::iterator start, typename std::vector<T>::iterator end,
                      MPI_Comm comm = MPI_COMM_WORLD) {

  auto minY = std::get<terminationFlagLayer>(*(std::min_element(start, end, layer_comparator<terminationFlagLayer, T>())));
  auto globalMinY = minY;

  mxx::datatype<decltype(minY)> dt;
  MPI_Datatype mpi_dt = dt.type();

  MPI_Allreduce(&minY, &globalMinY, 1, mpi_dt, MPI_MIN, comm);
  return globalMinY > 0x1;
}



//Prints out all the tuples on console
//Not supposed to be used with large datasets
template <typename T>
void printTuples(std::vector<T>& localVector, MPI_Comm comm = MPI_COMM_WORLD)
{
  /// MPI rank within the communicator
  int rank, commsize;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &commsize);

  for(auto &eachTuple : localVector)
  {
    std::cout << rank << " : " << eachTuple << std::endl;
  }
}
//Prints out all the tuples on console
//Not supposed to be used with large datasets
template <uint8_t keyLayer, uint8_t valueLayer, typename T>
void writeTuples(std::vector<T>& localVector, std::string filename)
{
  std::ofstream ofs;
  ofs.open(filename);

  ofs << "kmer, partition" << std::endl;

  for(auto &eachTuple : localVector)
  {
    ofs << std::get<keyLayer>(eachTuple) << "," << std::get<valueLayer>(eachTuple) << std::endl;
  }

  ofs.close();
}


#endif
