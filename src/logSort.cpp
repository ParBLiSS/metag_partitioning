/**
 * @file    log_sort.cpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the de bruijn graph clustering in log(D_max) time.
 *          This file is only used for benchmarking partitioning algorithm in SC15. 
 *          For application use, please see src/getHistogram.cpp
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */
//Includes
#include <mpi.h>
#include <iostream>
#include <limits>

#include "configParam.hpp"
#include "argvparser.h"

//File includes from BLISS
#include <common/kmer.hpp>
#include <common/base_types.hpp>

//Own includes
#include "parallel_fastq_iterate.hpp"
#include "ccl.hpp"

#include <mxx/timer.hpp>

#include <sstream>
#include "read_graph_utils.hpp"

using namespace std;
using namespace CommandLineProcessing;

int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  // get communicaiton parameters
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //Parse command line arguments
  ArgvParser cmd;
  cmdLineParams cmdLineVals;

  cmd.setIntroductoryDescription("Parallel partitioning algorithm used for benchmarking (SC15)");
  cmd.setHelpOption("h", "help", "Print this help page");

  cmd.defineOption("file", "Name of the dataset in the FASTQ format", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("method", "Type of log-sort to run (standard[Naive], inactive[AP], loadbalance[AP_LB])", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);

  int result = cmd.parse(argc, argv);

  if (result != ArgvParser::NoParserError)
  {
    if (!rank) cout << cmd.parseErrorDescription(result) << "\n";
    exit(1);
  }

  cmdLineVals.fileName = cmd.optionValue("file");
  cmdLineVals.method = cmd.optionValue("method"); 


  mxx::timer t;

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  //Specify Kmer Type
  const int kmerLength = KMER_LEN;
  typedef bliss::common::DNA AlphabetType;
  typedef bliss::common::Kmer<kmerLength, AlphabetType, uint64_t> KmerType;


  //Define tuple type
  typedef typename std::tuple<KmerIdType, PidType, PidType> tuple_t;


  MP_TIMER_START();

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;

  //Read all the kmers without any filter

  std::vector<bool> readFilterFlags;
  std::vector<ReadLenType> readTrimLengths;

  //Read all the kmers without any filter
  readFASTQFile< KmerType, includeAllKmersinAllReads<KmerType> > (cmdLineVals, localVector, readFilterFlags, readTrimLengths, MPI_COMM_WORLD);
  MP_TIMER_END_SECTION("Generating Data");


  ReadGraphGenerator::generate(localVector, MPI_COMM_WORLD);
  MP_TIMER_END_SECTION("Generating Read Graph");


  if (cmdLineVals.method == "standard")
    cluster_reads_par(localVector, MPI_COMM_WORLD);
  else if (cmdLineVals.method == "inactive")
    cluster_reads_par_inactive(false, localVector, MPI_COMM_WORLD);
  else if (cmdLineVals.method == "loadbalance")
    cluster_reads_par_inactive(true, localVector, MPI_COMM_WORLD);
  else {
    std::cout << "Usage: mpirun -np 4 <executable> --method <method> --file FASTQ_FILE\n";
    std::cout << "  where <method> can be: \"standard\" (Naive), \"inactive\"(AP) ,\"loadbalance\"(AP_LB)\n";
    return 1;
  }

  MPI_Finalize();
  return(0);
}


