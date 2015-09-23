#ifndef CONFIGPATH_HPP
#define CONFIGPATH_HPP


/*
 * ADJUSTABLE BY USER
 */

//Should be modified based on the system where the software is being executed


//Local disk space on each node in the distributed system for velvet (Multiple writes and reads)
//This path should be a valid directory on the execution node
const std::string localFS = "/local/scratch/cjain7/";

//Shared space for communicating reads across ranks (Used for only one time write and read)
//This path should be a valid directory on the execution node
const std::string sharedFS = "/lustre/alurugroup/Chirag/Metagenome_Data/tmpDirForCode"; 

//Absolute path to the project directory
//We need to access source code folder during execution
const std::string projSrcDir = "/work/alurugroup/chirag/Metagenomics/Partitioning/metag_partitioning/";

#endif
