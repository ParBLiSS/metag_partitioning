#ifndef POST_PROCESS_HPP 
#define POST_PROCESS_HPP 

//Includes
#include <mpi.h>

//Includes from mxx library
#include <mxx/sort.hpp>

//Own includes
#include "sortTuples.hpp"
#include "configParam.hpp"
#include "packedRead.hpp"

/*
 * @brief     Takes the kmer-pid tuples as input and spits out read-pid tuples as output
 * @details   
 *            1. Parse the file again to generate first_kmer-MAX-readId tuples and append to original vector
 *            2. Sort the vector by kmers and for each MAX tuple, append the new tuples with readId-Pid to a new vector
 *            3. The format of new tuples should be emptyread-readid-pid-0
 *            4. delete the original localVector, and return the new tuples 
 */
template <typename KmerType, typename T, typename Q>
void generateReadToPartitionMapping(  cmdLineParams& cmdLineVals,
                                      std::vector<T>& localVector, std::vector<Q>& newlocalVector, 
                                      std::vector<bool>& readFilterFlags,
                                      MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  //temporary vector for appending reads (first kmers) to localVector
  std::vector<T> localVector2;

  //Parse the first kmers of each read and keep in localVector2
  readFASTQFile< KmerType, includeFirstKmersinFilteredReads<KmerType> > (cmdLineVals, localVector2, readFilterFlags);

  //Paste all the values into original vector
  localVector.insert(localVector.end(), localVector2.begin(), localVector2.end());
  localVector2.clear();

  static layer_comparator<kmerTuple::kmer, T> kmerCmp;

  //Sort by kmer Layer
  mxx::block_decompose(localVector, comm);
  mxx::sort(localVector.begin(), localVector.end(), kmerCmp, comm, true); 

  //First resolve the first and last buckets
  //Vector for containing tuples with pids of leftmost and rightmost buckets
  std::vector<T> kmerIdsToSend(2);
  std::vector<T> kmerIdsReceived(2);

  //Disable the effect of these entries by default by setting MAX in the PnLayer 
  //Rationale: This rank might not contain valid Pid for boundary buckets
  std::get<kmerTuple::Pn>(kmerIdsToSend[0]) = MAX;  std::get<kmerTuple::Pn>(kmerIdsToSend[1]) = MAX;
  

  //First bucket's range
  auto firstBucketBound = findRange(localVector.begin(), localVector.end(), *(localVector.begin()), kmerCmp);

  for(auto it = firstBucketBound.first; it != firstBucketBound.second; it++)
  {
    //If there is a tuple containing the valid partition id for this kmer
    if(std::get<kmerTuple::Pn>(*it) != MAX)
    {
      //Save this tuple
      kmerIdsToSend[0] = *it;
    }
  }

  //Repeat the same process with rightmost bucket
  auto lastBucketBound = findRange(localVector.rbegin(), localVector.rend(), *(localVector.rbegin()), kmerCmp);
  for(auto it = lastBucketBound.first; it != lastBucketBound.second; it++)
  {
    //If there is a tuple containing the valid partition id for this kmer
    if(std::get<kmerTuple::Pn>(*it) != MAX)
    {
      //Save this tuple
      kmerIdsToSend[1] = *it;
    }
  }

  //Do the communication and get the partitionids for boundary partitions
  //We do the MPI exscan on rightMostKmersIdSend to get the ids for our bucket, higher rank 

  std::vector<T> gatherkmerIds = mxx::allgatherv(kmerIdsToSend, comm);

  //Fetch the correct leftmost and rightmost ids using the values gathered above
  kmerIdsReceived[0] = *(std::find_if(gatherkmerIds.begin(), gatherkmerIds.end(), 
                                                                              [&localVector = localVector](const T& x){
                                                                                return (std::get<kmerTuple::kmer>(x) == std::get<kmerTuple::kmer>(localVector.front()) &&
                                                                                        std::get<kmerTuple::Pn>(x) != MAX);}));
  kmerIdsReceived[1] = *(std::find_if(gatherkmerIds.begin(), gatherkmerIds.end(), 
                                                                              [&localVector = localVector](const T& x){
                                                                                return (std::get<kmerTuple::kmer>(x) == std::get<kmerTuple::kmer>(localVector.back()) &&
                                                                                        std::get<kmerTuple::Pn>(x) != MAX);}));

  //Now we are ready to go through each bucket
  //For each bucket, we first find the correct pid, then for each read tuple, make new tuple in newlocalVector with appropriate values

  for(auto it = localVector.begin(); it != localVector.end();)
  {
    auto innerLoopBound = findRange(it, localVector.end(), *it, kmerCmp);
    
    if (innerLoopBound.first == localVector.begin()) // first bucket
    {
      //Get the pid from boundary tuple
      auto correctPid = std::get<kmerTuple::Pc>(kmerIdsReceived[0]);

      //Scan the bucket
      for(auto it2 = innerLoopBound.first; it2!= innerLoopBound.second; it2++)
      {
        if(std::get<kmerTuple::Pn>(*it2) == MAX)
        {
          Q newTuple;
          std::get<readTuple::pid>(newTuple) = correctPid;
          std::get<readTuple::rid>(newTuple) = std::get<kmerTuple::Pc>(*it2);
          std::get<readTuple::cnt>(newTuple) = 0;

          //Push this tuple to new vector
          newlocalVector.push_back(newTuple);
        }
      }
    }
    else if (innerLoopBound.second == localVector.end()) // last bucket
    {
      //Get the pid from boundary tuple
      auto correctPid = std::get<kmerTuple::Pc>(kmerIdsReceived[1]);

      //Scan the bucket
      for(auto it2 = innerLoopBound.first; it2!= innerLoopBound.second; it2++)
      {
        if(std::get<kmerTuple::Pn>(*it2) == MAX)
        {
          Q newTuple;
          std::get<readTuple::pid>(newTuple) = correctPid;
          std::get<readTuple::rid>(newTuple) = std::get<kmerTuple::Pc>(*it2);
          std::get<readTuple::cnt>(newTuple) = 0;

          //Push this tuple to new vector
          newlocalVector.push_back(newTuple);
        }
      }
    }
    else if (innerLoopBound.first != localVector.begin() && innerLoopBound.second != localVector.end()) // middle bucket
    {
      //Get the pid from boundary tuple
      auto findTuple = *(std::find_if(innerLoopBound.first, innerLoopBound.second, 
                                                                              [](const T& x){
                                                                                return (std::get<kmerTuple::Pn>(x) != MAX);}));

      auto correctPid = std::get<kmerTuple::Pc>(findTuple);

      //Scan the bucket
      for(auto it2 = innerLoopBound.first; it2!= innerLoopBound.second; it2++)
      {
        if(std::get<kmerTuple::Pn>(*it2) == MAX)
        {
          Q newTuple;
          std::get<readTuple::pid>(newTuple) = correctPid;
          std::get<readTuple::rid>(newTuple) = std::get<kmerTuple::Pc>(*it2);
          std::get<readTuple::cnt>(newTuple) = 0;

          //Push this tuple to new vector
          newlocalVector.push_back(newTuple);
        }
      }
    }

    //Move to next bucket
    it = innerLoopBound.second;
  }
}

/*
 * @brief     Given readid-pid as input, this generates vector of tuples with read string sequences and partition id
 * @details
 *            1.  Parse all the read sequences and append new tuples for them, with pid set to MAX in the beginning
 *            2.  Sort the vector by readid, and assign pid to reads by linear scan in each bucket
 *            4.  Convert this vector to a vector having tuples with read sequences and correct pids. The vector should be
 *                sorted by pid in the end
 */
template <typename KmerType, typename Q> 
void generateSequencesVector(cmdLineParams& cmdLineVals,
                             std::vector<Q>& newLocalVector, std::vector<bool>& readFilterFlags,
                             MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  //temporary vector for appending reads (first kmers) to localVector
  std::vector<Q> newLocalVector2;

  //Parse the first kmers of each read and keep in localVector2
  readFASTQFile< KmerType, includeWholeReadinFilteredReads<KmerType> > (cmdLineVals, newLocalVector2, readFilterFlags);

  //Paste all the values into original vector
  newLocalVector.insert(newLocalVector.end(), newLocalVector2.begin(), newLocalVector2.end());
  newLocalVector2.clear();

  static layer_comparator<readTuple::rid, Q> ridCmp;

  //Sort by read id Layer
  mxx::block_decompose(newLocalVector, comm);
  mxx::sort(newLocalVector.begin(), newLocalVector.end(), ridCmp, comm, true); 

  //Now each bucket will have only 2 elements, read tuple with pid and read tuple with sequence.
  //Goal is to place the sequence in the tuple with correct pid and later remove tuples without pid
  //Since a bucket could be spilled across processors, we will do a left shift of leftmost element, and push_back it to local vector
  //Then each bucket is processed only when its size is 2

  Q tupleToShift = newLocalVector.front();
  Q tupleFromShift = mxx::left_shift(tupleToShift, comm);

  //Last but one processor
  if(rank < p - 1) 
    newLocalVector.push_back(tupleFromShift);

  //Start processing the buckets
  for(auto it = newLocalVector.begin(); it != newLocalVector.end();)
  {
    auto innerLoopBound = findRange(it, newLocalVector.end(), *it, ridCmp);

    if(innerLoopBound.second - innerLoopBound.first == 2)
    {
      //Get the pid from boundary tuple
      auto findTupleWithSequence = (std::find_if(innerLoopBound.first, innerLoopBound.second, 
            [](const Q& x){
            return (std::get<readTuple::pid>(x) == MAX);}));

      auto findTupleWithPid = (std::find_if(innerLoopBound.first, innerLoopBound.second,
            [](const Q& x){
            return (std::get<readTuple::pid>(x) != MAX);}));

      //Update sequence and count
      std::get<readTuple::seq>(*findTupleWithPid) = std::get<readTuple::seq>(*findTupleWithSequence);
      std::get<readTuple::cnt>(*findTupleWithPid) = std::get<readTuple::cnt>(*findTupleWithSequence);

    }
    else
    {
      //This tuple is not needed, therefore should be removed later
      std::get<readTuple::pid>(*innerLoopBound.first) = MAX;
    }

    //Move to next bucket
    it = innerLoopBound.second;
  }

  //Now we can get rid of all the tuples who have pid as MAX
  auto cend = std::partition(newLocalVector.begin(), newLocalVector.end(), 
      [](const Q& x){
      return std::get<readTuple::pid>(x) != MAX;});

  newLocalVector.erase(cend, newLocalVector.end());

  //Sort by partitionId to bring reads in same Pc adjacent
  static layer_comparator<readTuple::pid, Q> pidCmp;
  mxx::block_decompose(newLocalVector, comm);
  mxx::sort(newLocalVector.begin(), newLocalVector.end(), pidCmp, comm, true); 

}

/*
 * @brief                         At the moment, partition with smaller ids tend to be larger.
 *                                To resolve this issue, partition ids are shuffled using XOR function
 * @param[in/out] localVector     Accepts vector which is block decomposed, and returns the vector sorted
 *            
 */
template <typename Q> 
void shuffleAndResortPids(std::vector<Q> &localVector, MPI_Comm comm = MPI_COMM_WORLD)
{
  std::for_each(localVector.begin(), localVector.end(), 
      [](Q &t){ 
          //Source : http://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
          PidType x = std::get<readTuple::pid>(t);
          x = ((x >> 16) ^ x) * 0x45d9f3b;
          x = ((x >> 16) ^ x) * 0x45d9f3b;
          x = ((x >> 16) ^ x);
          std::get<readTuple::pid>(t) = x;
      });

  //Sort by partitionId to bring reads in same Pc adjacent
  static layer_comparator<readTuple::pid, Q> pidCmp;
  mxx::sort(localVector.begin(), localVector.end(), pidCmp, comm, true);
}

/*
 * @brief   A separate struct to initialize all the string commands needed to execute velvet assembler.
 *          Final output contigs are saved in the contigs.fa file 
 */
struct AssemblyCommands
{
  int rank;
  std::string filename_fasta, filename_contigs, outputDir, sharedFolder,
              cmd_create_dir, velvetExe1, velvetExe2,
              backupContigs, 
              resetVelvet, resetInput, resetOutput, resetSharedFolder,
              resetContigTmps,
              finalMerge;

  //Constructor
  AssemblyCommands(int rank_, cmdLineParams &cmdLineVals)
  {
    rank = rank_;
    do_init(cmdLineVals);
  }

  void do_init(cmdLineParams &cmdLineVals)
  {
    //filename for writing read sequences
    filename_fasta = localFS + "/reads_" + std::to_string(rank) + ".fasta";

    //Velvet writes it output to a directory, contigs are saved in contigs.fa file
    //Need to append those contigs in this rank's main contig file
    filename_contigs = localFS + "/contigs_" + std::to_string(rank) + ".fasta";

    //Output directory for this rank
    outputDir = localFS + "/velvetOutput_" + std::to_string(rank);

    //Shared folder to save reads in the boundary partition
    sharedFolder = sharedFS + "/tmpFolder";
    cmd_create_dir = "mkdir -p " + outputDir + " " + sharedFolder;

    //Executing assembler
    velvetExe1 = "velveth " + outputDir + " "  + std::to_string(cmdLineVals.velvetKmerSize) +" -short " + filename_fasta + " ";
    velvetExe2 = "velvetg " + outputDir + " ";

    //Assembler's output in the contigs file
    backupContigs = "cat " + outputDir + "/contigs.fa >> " + filename_contigs;

    //Remove file/directory after every assemby run
    resetVelvet = "rm -rf " + outputDir + "/*";
    resetInput = "> " + filename_fasta;
    resetContigTmps = "> " + filename_contigs;
    resetOutput = "rm -rf contigs.fa";
    resetSharedFolder = "rm -rf " + sharedFolder;

    //Concatenate all the contigs to a single file
    //Every rank should call this one by one
    finalMerge = "cat " + filename_contigs +" >> contigs.fa"; 
  }

  std::string getBoundaryFastaFileName(int rankWhereToSendReads)
  {
    return sharedFolder + "/reads_" + std::to_string(rankWhereToSendReads) + "_" + std::to_string(rank);
  }

  std::string catBoundaryFiles(int rankOwner)
  {
    std::string files = sharedFolder + "/reads_" + std::to_string(rankOwner) + "_*";
    return "if ls " + files + " 1> /dev/null 2>&1 ; then cat " + files + " >> " + filename_fasta + "; fi ;"; 
  }
};

/*
 * @brief     With read sequences and pids in the vector (sorted by pid), run parallel assembly using a assembler
 * @details
 *            1. Iterate over the element of vectors and dump the reads belonging to same partition into fasta file
 *            2. Run assembler and append the contigs to a file local to this processor
 *            3. Merge all the contig files in the end
 */
template <typename ReadInf, typename Q>
void runParallelAssembly(std::vector<Q> &localVector, cmdLineParams &cmdLineVals, MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  //Timer to see balance of assembly load among the ranks
  MP_TIMER_START();

  //Assuming localVector was sorted by pids in the previous step

  //The role of this function is trivial, except that there might be partitions spawning
  //across multiple ranks. To deal with this, we will handle end boundary partitions as special case
  
  //Comparator for computing a partition's range
  static layer_comparator<readTuple::pid, Q> pidCmp;

  AssemblyCommands R(rank, cmdLineVals);
  //Clean things in case output already exists
  int i;
  i = std::system(R.resetVelvet.c_str());
  i = std::system(R.resetInput.c_str());
  i = std::system(R.resetContigTmps.c_str());
  if(!rank) i = std::system(R.resetOutput.c_str());
  if(!rank) i = std::system(R.resetSharedFolder.c_str());

  //If output directory doesn't exist, create one
  i = std::system(R.cmd_create_dir.c_str());

  /*
   * NEED TO DO SOME WORK FOR RESOLVING BOUNDARY PARTITIONS
   * 1. Partition that spans over more than 1 rank, we will assume the highest rank owns it
   * 2. Let other ranks who don't own a partition write the reads to a separate file in sharedFolder
   * 3. Rank which owns the partition would read the partitions and pipe them to its own read fasta file
   */

  //Ids for first and last partition this rank has
  std::vector<PidType> boundaryPartitionIds(2);
  boundaryPartitionIds[0] = std::get<readTuple::pid>(localVector.front());
  boundaryPartitionIds[1] = std::get<readTuple::pid>(localVector.back());

  auto allBoundaryPartitionIds = mxx::allgatherv(boundaryPartitionIds, comm);
  
  //Decide if I owe someone else my last partition
  bool iShouldTransferLastPartition;
  int rankToWhom = MAX_INT;
  //IF : My last partition's Id matches with first partition's id of rank one higher to me
  if(rank < p - 1 && (allBoundaryPartitionIds[2*rank + 1] == allBoundaryPartitionIds[2*(rank + 1)]) )
  {
    iShouldTransferLastPartition = true;

    //Compute which rank to transfer 
    auto partitionRange = std::equal_range(allBoundaryPartitionIds.begin(), allBoundaryPartitionIds.end(), allBoundaryPartitionIds[2*rank + 1]);
    auto indexFromStartLastOccurence = std::distance(allBoundaryPartitionIds.begin(), partitionRange.second) - 1; 
    rankToWhom = ((int)indexFromStartLastOccurence)/2;
  }
  else
    iShouldTransferLastPartition = false;

  //See if I own my first partition or not
  bool iDontOwnPartitionFirstPartition;
  //IF :  I have only 1 partition with me and it needs to be transferred to higher rank
  if(allBoundaryPartitionIds[2*rank] == allBoundaryPartitionIds[2*rank + 1] && iShouldTransferLastPartition) 
    iDontOwnPartitionFirstPartition = true;
  else
    iDontOwnPartitionFirstPartition = false;

  //To write sequence to file
  std::ofstream ofs;

  //Save last partition to a file
  if(iShouldTransferLastPartition)
  {
    auto innerLoopBound = findRange(localVector.rbegin(), localVector.rend(), localVector.back(), pidCmp);
    ofs.open(R.getBoundaryFastaFileName(rankToWhom),std::ofstream::out);

    for(auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; it2++)
    {
      std::string seqHeader = ">" + std::to_string(std::get<readTuple::rid>(*it2)); 
      std::string seqString;
      getUnPackedRead<ReadInf>(std::get<readTuple::seq>(*it2), std::get<readTuple::cnt>(*it2), seqString);

      ofs << seqHeader << "\n" << seqString << "\n"; 
    }
    ofs.close();
  }

  //All boundary partitions should be ready in the file system before moving ahead
  MP_TIMER_END_SECTION("[ASSEMBLY TIMER] Boundary partitions ready on disk");
  MPI_Barrier(comm);

  for(auto it=localVector.begin(); it!=localVector.end(); )
  {
    auto innerLoopBound = findRange(it, localVector.end(), *it, pidCmp);
    ofs.open(R.filename_fasta,std::ofstream::out);

    bool runVelvet = false;

    //If this is first partition
    if(innerLoopBound.first == localVector.begin())
    {
      //Don't check the partition size here, because we are also getting reads from other ranks
      if(iDontOwnPartitionFirstPartition == false)
      {
        for(auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; it2++)
        {
          std::string seqHeader = ">" + std::to_string(std::get<readTuple::rid>(*it2)); 
          std::string seqString;
          getUnPackedRead<ReadInf>(std::get<readTuple::seq>(*it2), std::get<readTuple::cnt>(*it2), seqString);
          ofs << seqHeader << "\n" << seqString << "\n"; 
        }

        //Get reads from other ranks
        i = std::system(R.catBoundaryFiles(rank).c_str());
        runVelvet = true;
      }
    }
    else if(innerLoopBound.second == localVector.end())
    {
      if(iShouldTransferLastPartition == false && innerLoopBound.second - innerLoopBound.first >= MIN_READ_COUNT_FOR_ASSEMBLY)
      {  
        for(auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; it2++)
        {
          std::string seqHeader = ">" + std::to_string(std::get<readTuple::rid>(*it2)); 
          std::string seqString;
          getUnPackedRead<ReadInf>(std::get<readTuple::seq>(*it2), std::get<readTuple::cnt>(*it2), seqString);
          ofs << seqHeader << "\n" << seqString << "\n"; 
        }

        runVelvet = true;
      }
    }
    else if(innerLoopBound.second - innerLoopBound.first >= MIN_READ_COUNT_FOR_ASSEMBLY)
    {
      for(auto it2 = innerLoopBound.first; it2 != innerLoopBound.second; it2++)
      {
        std::string seqHeader = ">" + std::to_string(std::get<readTuple::rid>(*it2)); 
        std::string seqString;
        getUnPackedRead<ReadInf>(std::get<readTuple::seq>(*it2), std::get<readTuple::cnt>(*it2), seqString);
        ofs << seqHeader << "\n" << seqString << "\n"; 
      }
      runVelvet = true;
    }

    ofs.close();

    if(runVelvet)
    {
      i = std::system(R.velvetExe1.c_str());
      i = std::system(R.velvetExe2.c_str());
      i = std::system(R.backupContigs.c_str());
      i = std::system(R.resetVelvet.c_str());
      i = std::system(R.resetInput.c_str());
    }

    //Increment the loop variable
    it = innerLoopBound.second;
  }

  MP_TIMER_END_SECTION("[ASSEMBLY TIMER] Parallel assembly completed");
  MPI_Barrier(comm);
  for(int I = 0; I < p; I++)
  {
    if(I == rank)
    {
      i = std::system(R.finalMerge.c_str());
    }
    MPI_Barrier(comm);
  }


  //Delete shared folder to keep next run correct
  //Important otherwise files may corrupt future runs
  if(!rank) i = std::system(R.resetSharedFolder.c_str());

  //No-op
  i = i;
}

//Wrapper for all the post processing functions
template <typename KmerType,  typename T>
void finalPostProcessing(std::vector<T>& localVector, std::vector<bool>& readFilterFlags, cmdLineParams &cmdLineVals,
                        MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  //Determining storage container for read sequences
  typedef readStorageInfo<typename KmerType::KmerAlphabet, typename KmerType::KmerWordType> ReadSeqTypeInfo; 

  //Type of array to hold read sequence
  typedef std::array<typename ReadSeqTypeInfo::ReadWordType, ReadSeqTypeInfo::nWords> ReadSeqType;

  //tuple of type <Sequence, ReadId, PartitionId, Size of read> 
  //This order is defined in configParam.hpp as <seq,rid,pid,cnt>
  typedef std::tuple<ReadSeqType, ReadIdType, ReadIdType, uint32_t> tuple_t;

  //New vector type needs to be defined to hold read sequences
  std::vector<tuple_t> newlocalVector;

  //Get the newlocalVector populated with readId and partitionIds
  MP_TIMER_START();
  generateReadToPartitionMapping<KmerType>(cmdLineVals, localVector, newlocalVector, readFilterFlags);
  MP_TIMER_END_SECTION("[POSTPROCESS TIMER] ReadId-Pid mapping completed");

  //Get the newlocalVector poulated with vector of read strings and partition ids
  generateSequencesVector<KmerType>(cmdLineVals, newlocalVector, readFilterFlags);
  MP_TIMER_END_SECTION("[POSTPROCESS TIMER] ReadStrings-Pid mapping completed");

  //Logging the histogram of partition size in terms of reads
  std::string histFileName = "partitionRead.hist";
  if(!rank) std::cout << "Generating read histogram in file " << histFileName << "\n";
  generatePartitionSizeHistogram<readTuple::pid>(newlocalVector, histFileName);
  MP_TIMER_END_SECTION("[POSTPROCESS TIMER] Read sized partition histogram generated");

  shuffleAndResortPids(newlocalVector);
  MP_TIMER_END_SECTION("[POSTPROCESS TIMER] Partition ids shuffled");

  //Run parallel assembly
  runParallelAssembly<ReadSeqTypeInfo>(newlocalVector, cmdLineVals);
  MP_TIMER_END_SECTION("[POSTPROCESS TIMER] Parallel assembly completed");

}

#endif
