#ifndef SORT_TUPLES_HPP 
#define SORT_TUPLES_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp> 
#include <mxx/shift.hpp> 

//Own includes
#include "prettyprint.hpp"

static bool dummyBool;

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
void sortTuples(std::vector<T>& localVector, bool& wasSortLayerUpdated = dummyBool, MPI_Comm comm = MPI_COMM_WORLD)
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
    wasSortLayerUpdated = false;

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
          wasSortLayerUpdated = true;
          std::get<sortLayer>(*it2) = currentMinimum;
        }
      }
    }

    //Forward the outer iterator to beginning of next bucket
    it = innerLoopBound.second;
  }


  MPI_Barrier(comm);

  /***
   * NOW WE DEAL WITH STUPID BOUNDARY VALUES 
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
            wasSortLayerUpdated = true;
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
            wasSortLayerUpdated = true;
            std::get<sortLayer>(*it) = std::get<pickMinLayer>(tupleFromRight);
          }
        }
      }
    }
  }
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

#endif
