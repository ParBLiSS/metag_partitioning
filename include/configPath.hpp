#ifndef CONFIGPATH_HPP
#define CONFIGPATH_HPP


/*
 * ADJUSTABLE BY USER
 */

//Should be modified based on the system where the software is being executed


//Local disk space on each node in the distributed system for velvet (Multiple writes and reads)
const std::string localFS = "/home/chirag/Documents/GRA/Metagenomics/tmpDir";

//Shared space for communicating reads across ranks (Used for only one time write and read)
const std::string sharedFS = "/home/chirag/Documents/GRA/Metagenomics/tmpDir"; 

//Absolute path to the project directory
const std::string projSrcDir = "/home/chirag/Documents/GRA/Metagenomics/Metagenomics/";

#endif
