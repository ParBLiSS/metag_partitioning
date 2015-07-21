/**
 * @file    log_sort.cpp
 * @ingroup group
 * @author  Tony Pan<tpan7@gatech.edu>
 * @brief   Implements the de bruijn graph clustering in log(D_max) time.
 *          This file is only used for benchmarking partitioning algorithm in SC15. 
 *          For application use, please see src/getHistogram.cpp
 *
 * Copyright (c) 2015 Georgia Institute of Technology. All Rights Reserved.
 */
//Includes
#include <cstdlib>  // atoi
#include <mpi.h>
#include <iostream>

//Own includes
#include "configParam.hpp"
#include "argvparser.h"

#include "graph500-utils.hpp"
#include "ccl.hpp"

#include <mxx/timer.hpp>

#include <sstream>

using namespace std;
using namespace CommandLineProcessing;

#include <vector>
#include <tuple>


int main(int argc, char** argv)
{
  // Initialize the MPI library:
  MPI_Init(&argc, &argv);

  // get communicaiton parameters
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //Parse command line arguments
  ArgvParser cmd;
  cmdLineParamsGraph500 cmdLineVals;

  cmd.setIntroductoryDescription("Parallel partitioning algorithm used for benchmarking (SC15)");
  cmd.setHelpOption("h", "help", "Print this help page");

  cmd.defineOption("scale", "scale of graph for Graph500 generator = log(num of vertices)", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("edgefactor", "average edge degree for vertex for Graph500 generator", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("method", "Type of log-sort to run (standard[Naive], inactive[AP], loadbalance[AP_LB])", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("seedfile", "file to write out the seed for each component.", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);

  int result = cmd.parse(argc, argv);

  if (result != ArgvParser::NoParserError)
  {
    if (!rank) cout << cmd.parseErrorDescription(result) << "\n";
    exit(1);
  }

  cmdLineVals.scale = atol(cmd.optionValue("scale").c_str());
  cmdLineVals.edgefactor = atol(cmd.optionValue("edgefactor").c_str());
  cmdLineVals.method = cmd.optionValue("method"); 
  cmdLineVals.seedFile = cmd.optionValue("seedfile");



  mxx::timer t;

  /*
   * Indices inside tuple will go like this:
   * 0 : KmerId
   * 1 : P_new
   * 2 : P_old
   */

  MP_TIMER_START();
  //Define tuple type
  typedef ::std::tuple<int64_t, int64_t, int64_t> tuple_t;

  // Populate localVector for each rank and return the vector with all the tuples
  std::vector<tuple_t> localVector;

  //Read all the kmers without any filter
  Graph500Generator::generate(cmdLineVals, localVector, MPI_COMM_WORLD);

//  dump_vector(localVector, MPI_COMM_WORLD, "logsort.g500.input");

  MP_TIMER_END_SECTION("Generating Data");

  ensure_undirected_and_self_looping(localVector);
  MP_TIMER_END_SECTION("Preprocess Data");



  if (cmdLineVals.method == "standard")
    cluster_reads_par(localVector, MPI_COMM_WORLD);
  else if (cmdLineVals.method == "inactive")
    cluster_reads_par_inactive(false, localVector, MPI_COMM_WORLD);
  else if (cmdLineVals.method == "loadbalance")
    cluster_reads_par_inactive(true, localVector, MPI_COMM_WORLD);
  else {
    std::cout << "Usage: mpirun -np 4 <executable> --method <method> --scale log_n_verts --edgefactor vert_degree --seedfile output_seed_file\n";
    std::cout << "  where <method> can be: \"standard\" (Naive), \"inactive\"(AP) ,\"loadbalance\"(AP_LB)\n";
    return 1;
  }

//  dump_vector(localVector, MPI_COMM_WORLD, "logsort.g500.ccl");
  

    // get the seeds,
    auto seeds = get_partition_seeds(localVector, MPI_COMM_WORLD);
    std::string seedfile = cmdLineVals.seedFile;
    seedfile += ".";
    seedfile += cmdLineVals.method;
    dump_seeds(seeds, MPI_COMM_WORLD, seedfile);

//  dump_vector(seeds, MPI_COMM_WORLD, "logsort.g500.seeds");

  MPI_Finalize();
  return(0);
}


