/**
 * @file    getHistogram.cpp
 * @ingroup group
 * @author  Chirag Jain <cjain7@gatech.edu>
 * @brief   Implements the complete parallel pipeline for digi_norm, paritioning and assembly.
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */

//Includes
#include <mpi.h>
#include <iostream>

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "configParam.hpp"
#include "sortTuples.hpp"
#include "parallel_fastq_iterate.hpp"
#include "utils.hpp"
#include "preProcess.hpp"
#include "postProcess.hpp"
#include "argvparser.h"

#include <mxx/collective.hpp>
#include <mxx/distribution.hpp>
#include <mxx/timer.hpp>

#include <sstream>

using namespace std;
using namespace CommandLineProcessing;


int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  // get communicaiton parameters
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  //Parse command line arguments
  ArgvParser cmd;
  cmdLineParams cmdLineVals;

  cmd.setIntroductoryDescription("Parallel metagenomic assembly software implemented by cjain7@gatech.edu");
  cmd.setHelpOption("h", "help", "Print this help page");

  cmd.defineOption("file", "Name of the dataset in the FASTQ format", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("velvetK", "Kmer length to pass while running velvet", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);

  //Global timer for calculating total time
  mxx::timer t;
  double startTime = t.elapsed();

  int result = cmd.parse(argc, argv);

  if (result != ArgvParser::NoParserError)
  {
    if (!rank) cout << cmd.parseErrorDescription(result) << "\n";
    exit(1);
  }


  cmdLineVals.fileName = cmd.optionValue("file");
  cmdLineVals.velvetKmerSize = std::stoi(cmd.optionValue("velvetK")); 


  if(!rank) {
    std::cout << "Runnning with " << p << " processors.\n"; 
    std::cout << "Filename : " <<  cmdLineVals.fileName << "\n"; 
    std::cout << "Velvet kmer size : " << cmdLineVals.velvetKmerSize << "\n";
  }

  /*
   * PREPROCESSING PHASE
   */
  //Specify Kmer Type
  const int kmerLength_pre = KMER_LEN_PRE;
  typedef bliss::common::DNA AlphabetType;
  typedef bliss::common::Kmer<kmerLength_pre, AlphabetType, KmerIdType> KmerType_pre;


  //Tuple type for storing kmers during the filtering phase
  typedef typename std::tuple<KmerIdType, ReadIdType, KmerFreqType, KmerSNoType> tuple_t_pre;

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t_pre> localVector_pre;

  //Read filter flags to trim or remove reads. If bool value is true, we include complete read
  //If bool value if false, we check readTrimLengths[] to see if any partial length of read is set 
  //Based on the value of the length, we read the sequence partially or completely ignore it
  std::vector<bool> readFilterFlags;
  std::vector<ReadLenType> readTrimLengths;

  //Generate kmer tuples, keep filter off
  MP_TIMER_START();
  readFASTQFile< KmerType_pre, includeAllKmers<KmerType_pre> > (cmdLineVals, localVector_pre, readFilterFlags, readTrimLengths);
  MP_TIMER_END_SECTION("File read for pre-process");

  //Pre-process
  trimReadswithHighMedianOrMaxCoverage<>(localVector_pre, readFilterFlags, readTrimLengths);
  MP_TIMER_END_SECTION("Digital normalization plus High frequency trimming completed");

  //Delete the local vector
  localVector_pre.resize(0);

  //Initialize vector for partioning phase
  typedef typename std::tuple<KmerIdType, PidType, PidType> tuple_t;
  std::vector<tuple_t> localVector;

  //Specify Kmer Type
  const int kmerLength = KMER_LEN;
  typedef bliss::common::Kmer<kmerLength, AlphabetType, KmerIdType> KmerType;


  // define k-mer and operator types
  typedef KmerReduceAndMarkAsInactive<tuple_t> KmerReducerType;
  typedef PartitionReduceAndMarkAsInactive<tuple_t> PartitionReducerType;
  ActivePartitionPredicate<tuple_t> app;


  /*
   * IMPORTANT NOTE
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  // Populate localVector for each rank and return the vector with all the tuples
  readFASTQFile< KmerType, includeAllKmersinFilteredReads<KmerType> > (cmdLineVals, localVector, readFilterFlags, readTrimLengths);
  MP_TIMER_END_SECTION("File read for partitioning");


  // re-distirbute vector into equal block partition
  mxx::block_decompose(localVector);

  assert(localVector.size() > 0);
  auto start = localVector.begin();
  auto end = localVector.end();
  auto pend = end;


  //Sort tuples by KmerId
  bool keepGoing = true;
  int countIterations = 0;
  while (keepGoing) {

    // sort by k-mers and update Pn
    mxx::sort(start, pend, layer_comparator<kmerTuple::kmer, tuple_t>(), MPI_COMM_WORLD, true);
    KmerReducerType r1;
    r1(start, pend, MPI_COMM_WORLD);

    // sort by P_c and update P_c via P_n
    mxx::sort(start, pend, layer_comparator<kmerTuple::Pc, tuple_t>(), MPI_COMM_WORLD, true);
    PartitionReducerType r2;
    r2(start, pend, MPI_COMM_WORLD);

    // check for global termination
    keepGoing = !checkTermination<kmerTuple::Pn, tuple_t>(start, pend, MPI_COMM_WORLD);

    if (keepGoing) {
      // now reduce to only working with active partitions
      pend = std::partition(start, pend, app);
      // re-shuffle the partitions to counter-act the load-inbalance
      pend = mxx::block_decompose_partitions(start, pend, end);
    }

    countIterations++;
    if(!rank)
      std::cout << "[RANK 0] : Iteration # " << countIterations <<"\n";
  }

  //Lets ensure Pn and Pc are equal for every tuple
  //This was not ensured during the program run
  std::for_each(localVector.begin(), localVector.end(), [](tuple_t &t){ std::get<kmerTuple::Pn>(t) = std::get<kmerTuple::Pc>(t);});


  std::string histFileName = "partitionKmer.hist";

  if(!rank)
  {
    std::cout << "Algorithm took " << countIterations << " iteration.\n";
    std::cout << "Generating kmer histogram in file " << histFileName << "\n";
  }
  MP_TIMER_END_SECTION("Partitioning completed");

  generatePartitionSizeHistogram<kmerTuple::Pc>(localVector, histFileName);

  MP_TIMER_END_SECTION("Kmer Partition size histogram generated");

  finalPostProcessing<KmerType>(localVector, readFilterFlags, readTrimLengths, cmdLineVals);

  MPI_Barrier(MPI_COMM_WORLD);
  MP_TIMER_END_SECTION("Parallel assembly completed");

  double time = t.elapsed() - startTime;
  if(!rank)
  {
    std::cerr << "TOTAL time : " << time << " ms.\n";
  }

  MPI_Finalize();
  return(0);
}
