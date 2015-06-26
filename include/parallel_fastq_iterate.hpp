#ifndef PARALLEL_FASTQ_ITERATE_HPP 
#define PARALLEL_FASTQ_ITERATE_HPP 

//Includes
#include <mpi.h>


//File includes from BLISS
#include <io/fastq_loader.hpp>
#include <io/sequence_id_iterator.hpp>
#include <io/sequence_iterator.hpp>
#include <common/kmer.hpp>
#include <common/kmer_iterators.hpp>
#include <common/base_types.hpp>
#include <iterators/transform_iterator.hpp>
#include <utils/kmer_utils.hpp>

//Own includes
#include "configParam.hpp"
#include "packedRead.hpp"

//Includes from mxx library
#include <mxx/datatypes.hpp>
#include <mxx/shift.hpp>


/**
 * @details
 * Uses Bliss library to parse FASTQ file in parallel
 * Approach to build the vector
 * 1. Define file blocks and iterators for each rank
 * 2. Within each rank, iterate over all the reads
 *
 * @tparam T                      Type of elements in a vector to populate which should be std::vector of tuples
 * @tparam typeOfReadingOperation Determines what information we need from FASTQ reads 
 * @param[out] localVector        Reference of vector to populate
 * @note                          This function should be called by all MPI ranks
 */
template <typename KmerType, typename typeOfReadingOperation, typename T>
void readFASTQFile(     cmdLineParams &cmdLineVals,
                        std::vector<T>& localVector,
                        std::vector<bool>& readFilterFlags,
                        std::vector<ReadLenType>& readTrimLengths,
                        MPI_Comm comm = MPI_COMM_WORLD) 
{
  /// DEFINE file loader.  this only provides the L1 blocks, not reads.
  using FileLoaderType = bliss::io::FASTQLoader<CharType, false, true>; // raw data type :  use CharType    // TODO: change to true,false

  // from FileLoader type, get the block iter type and range type
  using FileBlockIterType = typename FileLoaderType::L1BlockType::iterator;

  /// DEFINE the iterator parser to get fastq records.  we don't need to parse the quality.
  using ParserType = bliss::io::FASTQParser<FileBlockIterType, void>;

  /// DEFINE the basic sequence type, derived from ParserType.
  using SeqType = typename ParserType::SequenceType;

  /// DEFINE the transform iterator type for parsing the FASTQ file into sequence records.
  using SeqIterType = bliss::io::SequencesIterator<ParserType>;

  //Get the Alphabet type
  using Alphabet = typename KmerType::KmerAlphabet;

  /// converter from ascii to alphabet values
  using BaseCharIterator = bliss::iterator::transform_iterator<typename SeqType::IteratorType, bliss::common::ASCII2<Alphabet> >;

  /// MPI rank within the communicator
  int rank;
  MPI_Comm_rank(comm, &rank);

  //==== create file Loader (single thread per MPI process)
  FileLoaderType loader(comm, cmdLineVals.fileName);  // this handle is alive through the entire building process.

  //====  now process the file, one L1 block (block partition by MPI Rank) at a time
  typename FileLoaderType::L1BlockType partition = loader.getNextL1Block();

  //Determines what information we need from FASTQ reads 
  typeOfReadingOperation vec;

  //We should reserve the space in the vector
  size_t read_size = loader.getRecordSize(5);
  size_t kmers_per_read = read_size / 2 - KmerType::size;
  size_t num_reads = (partition.getRange().size() + read_size - 1) / read_size ;
  size_t num_kmers = num_reads * kmers_per_read;

  vec.reserveSpace(localVector, num_kmers, num_reads);

  //Loop over all the L1 partitions

  ReadIdType readId = 0;
  while(partition.getRange().size() > 0)
  {

    //== process the chunk of data
    SeqType read;

    //==  and wrap the chunk inside an iterator that emits Reads.
    //== instantiate a local parser on each rank
    ParserType parser;

    SeqIterType seqs_start(parser, partition.begin(), partition.end(), partition.getRange().start);
    SeqIterType seqs_end(partition.end());

    //== loop over the reads
    for (; seqs_start != seqs_end; ++seqs_start)
    {
      // first get read
      read = *seqs_start;

      //Sanity check
      if (read.seqBegin == read.seqEnd) continue;

      //== transform ascii to coded value
      BaseCharIterator charStart(read.seqBegin, bliss::common::ASCII2<Alphabet>());
      BaseCharIterator charEnd(read.seqEnd, bliss::common::ASCII2<Alphabet>());


      //Parse the read and place the required values inside the vector
      vec.fillValuesfromReads(localVector, charStart, charEnd, readFilterFlags, readTrimLengths, readId); 

      readId += 1;
    }

    //Get next L1 block for this MPI process
    partition = loader.getNextL1Block();
  }

  MPI_Barrier(comm);

  //Need to make sure that Ids are unique across MPI procs
  vec.globalUniquenessOfIds(localVector, readId, comm);
  

  //For logging purpose, count total kmers across all the nodes
  auto localVecSize = localVector.size();

  //Global vector size
  uint64_t globalVecSize;
  MPI_Reduce(&localVecSize, &globalVecSize, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);

  if(rank == 0)
    std::cout << "Total count of tuples: " << globalVecSize << " \n";
}

/*
 * Generate a vector of tuples(kmer, Pn, Pc) from FASTQ file for each MPI process
 * Uses the tuples within  all the reads in the FASTQ file
 * Each Pn and Pc should by initialized with readIds
 */
template <typename KmerType>
struct includeAllKmers
{
  //Reserve space in the vector
  template <typename T>
  void reserveSpace(std::vector<T>& localVector, size_t num_kmers, size_t num_reads)
  {
    localVector.reserve(num_kmers * 1.1);
  }

  //Fill values in the localVector
  template <typename T, typename BaseCharIterator>
  void fillValuesfromReads(std::vector<T>& localVector, BaseCharIterator charStart, BaseCharIterator charEnd, 
                          std::vector<bool>& readFilterFlags, std::vector<ReadLenType>& readTrimLengths, 
                          ReadIdType readId)
  {
    /// kmer generation iterator
    using KmerIterType = bliss::common::KmerGenerationIterator<BaseCharIterator, KmerType>;

    //== set up the kmer generating iterators.
    KmerIterType start(charStart, true);
    KmerIterType end(charEnd, false);

    //Either the filter switch is off or if its own, flag should be true
    
    int kmerSerialNo = 0;
    for (; start != end; ++start)
    {
      //New tuple that goes inside the vector
      T tupleToInsert;

      //Current kmer
      auto originalKmer = *start;

      //Get the reverse complement
      auto reversedKmer = (*start).reverse_complement();

      //Choose the minimum of two to insert in the vector
      auto KmerToinsert = (originalKmer < reversedKmer) ? originalKmer : reversedKmer;

      //getPrefix() on kmer gives a 64-bit prefix for hashing 
      std::get<kmerTuple_Pre::kmer>(tupleToInsert) = (KmerToinsert).getPrefix();
      std::get<kmerTuple_Pre::rid>(tupleToInsert) = readId;
      std::get<kmerTuple_Pre::kmer_sno>(tupleToInsert) = kmerSerialNo++;

      //Insert tuple to vector
      localVector.push_back(tupleToInsert);
    }
  }

  //Maintain the global order of Ids across all MPI ranks
  template <typename T>
  void globalUniquenessOfIds(std::vector<T>& localVector, ReadIdType localReadCount, MPI_Comm comm)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    ReadIdType previousReadIdSum;

    //Get MPI Datatype using mxx library
    mxx::datatype<ReadIdType> MPI_ReadIDType;
    MPI_Exscan(&localReadCount, &previousReadIdSum, 1, MPI_ReadIDType.type(), MPI_SUM, comm);


    //Update all elements
    if(rank > 0)
    {
      for ( auto& eachTuple : localVector) 
      {
        //Update Pn and Pc
        std::get<kmerTuple_Pre::rid>(eachTuple) = std::get<kmerTuple_Pre::rid>(eachTuple) + previousReadIdSum;
      }
    }
  }

};

/*
 * Generate a vector of tuples(kmer, Pn, Pc) from FASTQ file for each MPI process
 * Ignore the reads which are discarded during preProcess stage
 * Each Pn and Pc should by initialized with readIds
 */
template <typename KmerType>
struct includeAllKmersinFilteredReads
{
  //Reserve space in the vector
  template <typename T>
  void reserveSpace(std::vector<T>& localVector, size_t num_kmers, size_t num_reads)
  {
    localVector.reserve(num_kmers * 1.1);
  }

  //Fill values in the localVector
  template <typename T, typename BaseCharIterator>
  void fillValuesfromReads(std::vector<T>& localVector, BaseCharIterator charStart, BaseCharIterator charEnd, 
                          std::vector<bool>& readFilterFlags, std::vector<ReadLenType>& readTrimLengths, 
                          ReadIdType readId)
  {
    /// kmer generation iterator
    using KmerIterType = bliss::common::KmerGenerationIterator<BaseCharIterator, KmerType>;

    //== set up the kmer generating iterators.
    KmerIterType start(charStart, true);
    KmerIterType end(charEnd, false);

    //Count how many kmers are we allowed to parse from this read
    int countKmersToRead = 0;
    if(readFilterFlags[readId] == true)
      countKmersToRead = MAX_INT;
    else if(readFilterFlags[readId] == false)
    {
      if(readTrimLengths[readId] == 0)
        countKmersToRead = 0;
      else
      {
        countKmersToRead = std::max(0, readTrimLengths[readId] - KMER_LEN + 1); 
      }
    }

    for (; start != end; ++start)
    {
      //Read filters affect here
      if(countKmersToRead-- == 0)
        break;

      //New tuple that goes inside the vector
      T tupleToInsert;

      //Current kmer
      auto originalKmer = *start;

      //Get the reverse complement
      auto reversedKmer = (*start).reverse_complement();

      //Choose the minimum of two to insert in the vector
      auto KmerToinsert = (originalKmer < reversedKmer) ? originalKmer : reversedKmer;

      //getPrefix() on kmer gives a 64-bit prefix for hashing 
      std::get<kmerTuple::kmer>(tupleToInsert) = (KmerToinsert).getPrefix();
      std::get<kmerTuple::Pn>(tupleToInsert) = readId;
      std::get<kmerTuple::Pc>(tupleToInsert) = readId;

      //Insert tuple to vector
      localVector.push_back(tupleToInsert);
    }
  }

  //Maintain the global order of Ids across all MPI ranks
  template <typename T>
  void globalUniquenessOfIds(std::vector<T>& localVector, ReadIdType localReadCount, MPI_Comm comm)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    ReadIdType previousReadIdSum;

    //Get MPI Datatype using mxx library
    mxx::datatype<ReadIdType> MPI_ReadIDType;
    MPI_Exscan(&localReadCount, &previousReadIdSum, 1, MPI_ReadIDType.type(), MPI_SUM, comm);


    //Update all elements
    if(rank > 0)
    {
      for ( auto& eachTuple : localVector) 
      {
        //Update Pn and Pc
        std::get<kmerTuple::Pn>(eachTuple) = std::get<kmerTuple::Pn>(eachTuple) + previousReadIdSum;
        std::get<kmerTuple::Pc>(eachTuple) = std::get<kmerTuple::Pn>(eachTuple);
      }
    }
  }
};

/*
 * Generate a vector of tuples(kmer, Pn, Pc) from FASTQ file for each MPI process
 * Ignore the reads which are discarded during preProcess stage
 * Each Pn and Pc should by initialized with readIds
 */
template <typename KmerType>
struct includeFirstKmersinFilteredReads
{
  //Reserve space in the vector
  template <typename T>
  void reserveSpace(std::vector<T>& localVector, size_t num_kmers, size_t num_reads)
  {
    localVector.reserve(num_reads);
  }

  //Fill values in the localVector
  template <typename T, typename BaseCharIterator>
  void fillValuesfromReads(std::vector<T>& localVector, BaseCharIterator charStart, BaseCharIterator charEnd, 
                          std::vector<bool>& readFilterFlags, std::vector<ReadLenType>& readTrimLengths, 
                          ReadIdType readId)
  {
    /// kmer generation iterator
    using KmerIterType = bliss::common::KmerGenerationIterator<BaseCharIterator, KmerType>;

    //== set up the kmer generating iterators.
    KmerIterType start(charStart, true);
    KmerIterType end(charEnd, false);

    //Count how many kmers are we allowed to parse from this read
    int countKmersToRead = 0;
    if(readFilterFlags[readId] == true)
      countKmersToRead = MAX_INT;
    else if(readFilterFlags[readId] == false)
    {
      if(readTrimLengths[readId] == 0)
        countKmersToRead = 0;
      else
      {
        countKmersToRead = std::max(0, readTrimLengths[readId] - KMER_LEN + 1); 
      }
    }

    //Either the filter switch is off or if its own, flag should be true
    if(countKmersToRead > 0)
    {
      //New tuple that goes inside the vector
      T tupleToInsert;

      //Current kmer
      auto originalKmer = *start;

      //Get the reverse complement
      auto reversedKmer = (*start).reverse_complement();

      //Choose the minimum of two to insert in the vector
      auto KmerToinsert = (originalKmer < reversedKmer) ? originalKmer : reversedKmer;

      //getPrefix() on kmer gives a 64-bit prefix for hashing 
      std::get<kmerTuple::kmer>(tupleToInsert) = (KmerToinsert).getPrefix();
      std::get<kmerTuple::Pn>(tupleToInsert) = MAX;
      std::get<kmerTuple::Pc>(tupleToInsert) = readId;

      //Insert tuple to vector
      localVector.push_back(tupleToInsert);
    }
  }

  //Maintain the global order of Ids across all MPI ranks
  template <typename T>
  void globalUniquenessOfIds(std::vector<T>& localVector, ReadIdType localReadCount, MPI_Comm comm)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    ReadIdType previousReadIdSum;

    //Get MPI Datatype using mxx library
    mxx::datatype<ReadIdType> MPI_ReadIDType;
    MPI_Exscan(&localReadCount, &previousReadIdSum, 1, MPI_ReadIDType.type(), MPI_SUM, comm);


    //Update all elements
    if(rank > 0)
    {
      for ( auto& eachTuple : localVector) 
      {
        //Update Pc only
        std::get<kmerTuple::Pc>(eachTuple) = std::get<kmerTuple::Pc>(eachTuple) + previousReadIdSum;
      }
    }
  }
};

template <typename KmerType>
struct includeWholeReadinFilteredReads
{
  //Reserve space in the vector
  template <typename Q>
  void reserveSpace(std::vector<Q>& localVector, size_t num_kmers, size_t num_reads)
  {
    localVector.reserve(num_reads);
  }

  //Fill values in the localVector
  template <typename Q, typename BaseCharIterator>
  void fillValuesfromReads(std::vector<Q>& localVector, BaseCharIterator charStart, BaseCharIterator charEnd, 
                          std::vector<bool>& readFilterFlags, std::vector<ReadLenType>& readTrimLengths, 
                          ReadIdType readId)
  {
    //Count how many kmers are we allowed to parse from this read
    int countCharsToRead = 0;
    if(readFilterFlags[readId] == true)
      countCharsToRead = charEnd - charStart;
    else if(readFilterFlags[readId] == false)
    {
      if(readTrimLengths[readId] == 0)
        countCharsToRead = 0;
      else
      {
        countCharsToRead = readTrimLengths[readId]; 
      }
    }

    //Either the filter switch is off or if its own, flag should be true
    if(countCharsToRead >= KMER_LEN)
    {
      //New tuple that goes inside the vector
      Q tupleToInsert;

      //Get the read in packed format using the input iterators
      typedef readStorageInfo <typename KmerType::KmerAlphabet, typename KmerType::KmerWordType> ReadInf;
      getPackedRead<ReadInf>(std::get<readTuple::seq>(tupleToInsert), charStart, charStart + countCharsToRead);

      //Fill other values in the tupleToInsert
      std::get<readTuple::rid>(tupleToInsert) = readId;
      std::get<readTuple::pid>(tupleToInsert) = MAX;
      std::get<readTuple::cnt>(tupleToInsert) = countCharsToRead;

      //Insert tuple to vector
      localVector.push_back(tupleToInsert);
    }
  }

  //Maintain the global order of Ids across all MPI ranks
  template <typename Q>
  void globalUniquenessOfIds(std::vector<Q>& localVector, ReadIdType localReadCount, MPI_Comm comm)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    ReadIdType previousReadIdSum;

    //Get MPI Datatype using mxx library
    mxx::datatype<ReadIdType> MPI_ReadIDType;
    MPI_Exscan(&localReadCount, &previousReadIdSum, 1, MPI_ReadIDType.type(), MPI_SUM, comm);


    //Update all elements
    if(rank > 0)
    {
      for ( auto& eachTuple : localVector) 
      {
        //Update Pc only
        std::get<readTuple::rid>(eachTuple) = std::get<readTuple::rid>(eachTuple) + previousReadIdSum;
      }
    }
  }


};


#endif
