// C/C++ includes
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <assert.h>

// File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

// include mxx sort
#include <mxx/sort.hpp>
#include <mxx/distribution.hpp>
// from external repository
#include <timer.hpp>

// own includes
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"

// section timer macros
#define MP_ENABLE_TIMER 1
#if MP_ENABLE_TIMER
#define MP_TIMER_START() TIMER_START()
#define MP_TIMER_END_SECTION(str) TIMER_END_SECTION(str)
#else
#define MP_TIMER_START()
#define MP_TIMER_END_SECTION(str)
#endif


int main(int argc, char** argv)
{
    // Initialize the MPI library:
    MPI_Init(&argc, &argv);

    //Specify the fileName
    std::string filename; 
    if( argc > 1 ) {
        filename = argv[1];
    } else {
        std::cout << "Usage: mpirun -np 4 <executable> FASTQ_FILE\n";
        return EXIT_FAILURE;
    }

    //Specify Kmer Type
    const int kmerLength = 31;
    typedef bliss::common::DNA AlphabetType;
    typedef bliss::common::Kmer<kmerLength, AlphabetType, uint64_t> KmerType;
    //Assuming kmer-length is less than 32
    typedef uint64_t KmerIdType;
    //Assuming read count is less than 4 Billion
    typedef uint32_t ReadIdType;
    /*
     * Indices inside tuple will go like this:
     * 0 : KmerId
     * 1 : P_new
     * 2 : P_old
     */
    typedef typename std::tuple<KmerIdType, ReadIdType, ReadIdType> tuple_t;
    typedef KmerReduceAndMarkAsInactive<0, 2, 1, tuple_t> KmerReducerType;
    typedef PartitionReduceAndMarkAsInactive<2, 1, tuple_t> PartitionReducerType;

    // get rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    if(!rank) {
        std::cout << "Runnning with " << p << " processors.\n"; 
        std::cout << "Filename : " <<  filename << "\n"; 
    }

    // start timer
    timer t;
    double startTime = t.get_ms();

    // read input file
    MP_TIMER_START();
    std::vector<tuple_t> localVector;
    //Populate localVector for each rank and return the vector with all the tuples
    generateReadKmerVector<KmerType, AlphabetType, ReadIdType> (filename, localVector, comm);
    MP_TIMER_END_SECTION("Read data from disk");

    // block partition vector
    mxx::block_decompose(localVector, comm);

    //Sort tuples by KmerId
    bool keepGoing = true;
    int countIterations = 0;

    // get vector iterators
    assert(localVector.size() > 0);
    auto start = localVector.begin();
    auto kend = localVector.end();
    auto pend = localVector.end();
    auto end = localVector.end();

    // initialize operators
    ActivePartitionPredicate<1, tuple_t> app;
    BoundaryKmerPredicate<1, tuple_t> bkp;
    layer_comparator<0, tuple_t> kmer_comp;
    layer_comparator<2, tuple_t> pc_comp;
    KmerReducerType kmer_reduc;
    PartitionReducerType part_reduc;


    while (keepGoing)
    {
        // sort by k-mer (only active k-mers)
        mxx::sort(start, kend, kmer_comp, comm, false);
        MP_TIMER_END_SECTION("mxx::sort by k-mer (active kmers)");

        // reduce through k-mer neighborhood
        kmer_reduc(start, kend, comm);
        MP_TIMER_END_SECTION("k-mer reduction (active kmers)");

        // sort by Pc (previous partition)
        mxx::sort(start, kend, pc_comp, comm, false);
        MP_TIMER_END_SECTION("mxx::sort by Pc (active kmers)");

        // now k-mers are back into their original partition

        // now we need to update the entire active partition including
        // interior kmers.  via a reduction, which requires a local sort first
        std::sort(start, pend, pc_comp);
        MP_TIMER_END_SECTION("local std::sort by Pc (all kmers)");

        // update partition (Pn -> Pc)
        part_reduc(start, pend, comm);
        MP_TIMER_END_SECTION("partition reduction Pn->Pc (all kmers)");

        // check for termination (all partitions are marked "inactive")
        keepGoing = !checkTermination<1, tuple_t>(start, pend, comm);
        MP_TIMER_END_SECTION("termination check");

        if (keepGoing) {
            // now the old active partitions are updated, but they could be scattered
            // to multiple processors.  some may have become inactive.
            // we need to keep interior and boundary kmers of an active partition
            // contiguously stored in the global array, so that in the next iteration
            // the boundary_kmer_sort - reduce - pc_sort sequence put the kmers
            // back to where they started from, so that interior kmers can be
            // updated.

            // remove inactive partitions (store at the end of the range)
            pend = std::partition(start, pend, app);
            MP_TIMER_END_SECTION("std::partition partitions (all kmers)");

            // re-shuffle the partitions to counter-act the load-inbalance
            pend = mxx::block_decompose_partitions(start, pend, end);
            MP_TIMER_END_SECTION("iteration mxx::block_decompose_partitions completed");

            // so we need to do a global sort by (new) pc now to keep all kmers in
            // active partitions together.  no reduction is needed.
            mxx::sort(start, pend, pc_comp, comm, true);
            MP_TIMER_END_SECTION("mxx::sort by Pc (all kmers)");

            // separate inactive k-mers to the end of the sequence
            kend = std::partition(start, pend, bkp);
            MP_TIMER_END_SECTION("std::partition kmers (all kmers)");
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

    // stop time and output
    double time = t.get_ms() - startTime;
    if(!rank) {
        std::cout << "Algorithm took " << countIterations << " iteration.\n";
        std::cout << "TOTAL TIME : " << time << " ms.\n";
    }

    // clean up
    MPI_Finalize();
    return(0);
}


