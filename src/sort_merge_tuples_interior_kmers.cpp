//Includes
#include <mpi.h>
#include <iostream> 

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"

//from external repository
#include <timer.hpp>

#include <sstream>

/*
 * Uses timer.hpp from Patrick's psack-copy
 * Couple of variables like p and rank should be defined as communicator size and MPI rank within the code
 */

#define MP_ENABLE_TIMER 1
#if MP_ENABLE_TIMER
#define MP_TIMER_START() TIMER_START()
#define MP_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#else
#define MP_TIMER_START()
#define MP_TIMER_END_SECTION(str)
#endif

//To output all the kmers and their respective partitionIds
//Switch on while testing


int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  //Specify the fileName
  std::string filename; 
  if( argc > 1 ) {
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

  MPI_Comm comm = MPI_COMM_WORLD;


  //Assuming kmer-length is less than 32
  typedef uint64_t KmerIdType;

  //Assuming read count is less than 4 Billion
  typedef uint32_t ReadIdType;

  //Know rank
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  if(!rank)
  {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  filename << "\n"; 
  }

  timer t;
  double startTime = t.get_ms();
  

  //Initialize the KmerVector
  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */
  typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;
  std::vector<tuple_t> localVector;

  typedef KmerReduceAndMarkAsInactive<0, 2, 1, tuple_t> KmerReducerType;
  typedef PartitionReduceAndMarkAsInactive<2, 1, tuple_t> PartitionReducerType;

  MP_TIMER_START();

  //Populate localVector for each rank and return the vector with all the tuples
  generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, comm);

  MP_TIMER_END_SECTION("Read data from disk");

//  printTuples<0, 2, tuple_t>(localVector.begin(), localVector.end(), MPI_COMM_WORLD);

  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;

  assert(localVector.size() > 0);

  auto start = localVector.begin();
  auto kend = localVector.end();
  auto pend = localVector.end();

  ActivePartitionPredicate<1, tuple_t> app;

  BoundaryKmerPredicate<1, tuple_t> bkp;

  layer_comparator<0, tuple_t> kmer_comp;
  layer_comparator<2, tuple_t> pc_comp;

  KmerReducerType kmer_reduc;
  PartitionReducerType part_reduc;


    while (keepGoing)
    {

      // take the boundary kmers and do joining via reduction
      {
        //MP_TIMER_START();
      //Sort the vector by each tuple's "sortLayer"th element
        mxx::sort(start, kend, kmer_comp, comm, false);
        //MP_TIMER_END_SECTION("    kmer mxx sort completed");
      }
      {
        //MP_TIMER_START();
        kmer_reduc(start, kend, comm);
        //MP_TIMER_END_SECTION("    kmer reduction completed");
      }
      // then put them back to the original nodes
      {
        //MP_TIMER_START();
      //Sort the vector by each tuple's "sortLayer"th element
        mxx::sort(start, kend, pc_comp, comm, false);
        //MP_TIMER_END_SECTION("    pold mxx sort completed");
      }
      // after this step, the updated boundary kmers are back in
      // their original partition via sort by Pc.

      // now we need to update the entire active partition including
      // interior kmers.  via a reduction, which requires a local sort first
      {
        //MP_TIMER_START();
        std::sort(start, pend, pc_comp);
        //MP_TIMER_END_SECTION("    part local sort completed");
      }
      // then do the reduction.
      {
        //MP_TIMER_START();
        part_reduc(start, pend, comm);
        //MP_TIMER_END_SECTION("    part reduction completed");
      }

      // at this point, we can check for termination.
      {
        //MP_TIMER_START();
        //Check whether all processors are done, and also update "pend"
        keepGoing = !checkTermination<1, tuple_t>(start, pend, comm);
        //MP_TIMER_END_SECTION("iteration Check phase completed");
      }

      if (keepGoing) {
        // now the old active partitions are updated, but they could be scattered
        // to multiple processors.  some may have become inactive.
        // we need to keep interior and boundary kmers of an active partition
        // contiguously stored in the global array, so that in the next iteration
        // the boundary_kmer_sort - reduce - pc_sort sequence put the kmers back to where
        // they started from, so that interior kmers can be updated.


        // we can allow the inactive partitions to be split between processes
        // but we need to make sure active partitions go on to the next iteration.
        pend = std::partition(start, pend, app);


        // so we need to do a global sort by (new) pc now to keep all kmers in active partitions together.
        // no reduction is needed.
        {
          //MP_TIMER_START();
          //Sort the vector by each tuple's "sortLayer"th element
          mxx::sort(start, pend, pc_comp, comm, false);
          //MP_TIMER_END_SECTION("    pnew mxx sort completed");
        }

        // and reduce working set further to boundary kmers
        kend = std::partition(start, pend, bkp);

      }
      countIterations++;
      if(!rank)
        std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";
    }


#if OUTPUTTOFILE
  //Output all (Kmer, PartitionIds) to a file in sorted order by Kmer
  //Don't play with the 0, 2 order, this is assumed by outputCompare
  if(!rank) std::cout << "WARNING: write to file option enabled \n";
  writeTuplesAll<0, 2, tuple_t>(localVector.begin(), localVector.end(), filename);
#endif


  double time = t.get_ms() - startTime;

    if(!rank)
    {
      std::cout << "Algorithm took " << countIterations << " iteration.\n";
      std::cout << "TOTAL TIME : " << time << " ms.\n"; 
    }

  MPI_Finalize();   
  return(0);
}


