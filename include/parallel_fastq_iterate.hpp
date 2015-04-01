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

//Own includes

//Includes from mxx library
#include <mxx/datatypes.hpp>
#include <mxx/shift.hpp>


/**
 * @details
 * Generate a vector of tuples(kmer, Pn, Pc) from FASTQ file for each MPI process
 * Each Pn and Pc should by initialized with readIds
 * Uses Tony's Bliss code
 * Approach to build the vector
 * 1. Define file blocks and iterators for each rank
 * 2. Within each rank, iterate over all the reads
 * 3. For each read, iterate over all the kmers in them and push them to vector
 * 4. Return the vector
 *
 * @tparam T                Type of vector to populate which should be std::vector of tuples
 * @param[out] localVector  Reference of vector to populate
 * @note                    This function should be called by all MPI ranks
 *                          No barrier at the end of the function execution
 */
template <typename KmerType, typename Alphabet, typename ReadIDType, typename T>
void generateReadKmerVector(const std::string &filename,
                        T& localVector,
                        MPI_Comm comm = MPI_COMM_WORLD) 
{
  /// DEFINE file loader.  this only provides the L1 blocks, not reads.
  using FileLoaderType = bliss::io::FASTQLoader<CharType, false, true>; // raw data type :  use CharType

  // from FileLoader type, get the block iter type and range type
  using FileBlockIterType = typename FileLoaderType::L2BlockType::iterator;

  /// DEFINE the iterator parser to get fastq records.  we don't need to parse the quality.
  using ParserType = bliss::io::FASTQParser<FileBlockIterType, void>;

  /// DEFINE the basic sequence type, derived from ParserType.
  using SeqType = typename ParserType::SequenceType;

  /// DEFINE the transform iterator type for parsing the FASTQ file into sequence records.
  using SeqIterType = bliss::io::SequencesIterator<ParserType>;

  /// converter from ascii to alphabet values
  using BaseCharIterator = bliss::iterator::transform_iterator<typename SeqType::IteratorType, bliss::common::ASCII2<Alphabet> >;

  /// kmer generation iterator
  typedef bliss::common::KmerGenerationIterator<BaseCharIterator, KmerType> KmerIterType;

  /// MPI rank within the communicator
  int rank;
  MPI_Comm_rank(comm, &rank);

  //==== create file Loader (single thread per MPI process)
  FileLoaderType loader(comm, filename);  // this handle is alive through the entire building process.

  //====  now process the file, one L1 block (block partition by MPI Rank) at a time
  typename FileLoaderType::L1BlockType partition = loader.getNextL1Block();

  //Loop over all the L1 partitions

  ReadIDType readId = 0;
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

      //Generate kmers out of the read and push to vector

      //Sanity check
      if (read.seqBegin == read.seqEnd) continue;

      //== transform ascii to coded value
      BaseCharIterator charStart(read.seqBegin, bliss::common::ASCII2<Alphabet>());
      BaseCharIterator charEnd(read.seqEnd, bliss::common::ASCII2<Alphabet>());

      //== set up the kmer generating iterators.
      KmerIterType start(charStart, true);
      KmerIterType end(charEnd, false);

      // NOTE: need to call *start to actually evaluate.  question is whether ++ should be doing computation.
      for (; start != end; ++start)
      {
        //Make tuple
        //getPrefix() on kmer gives a 64-bit prefix for hashing assuming 
        auto tupleToInsert = std::make_tuple((*start).getPrefix(), readId, readId);

        //Insert tuple to vector
        localVector.push_back(tupleToInsert);
      }

      readId += 1;
    }

    partition = loader.getNextL1Block();
  }

  //In order to make sure the readIds are unique across MPI procs, we need to communicate and update the local
  //maximum read id
  MPI_Barrier(comm);

  ReadIDType previousReadIdSum;

  //Get MPI Datatype using mxx library
  mxx::datatype<ReadIDType> MPI_ReadIDType;
  MPI_Exscan(&readId, &previousReadIdSum, 1, MPI_ReadIDType.type(), MPI_SUM, comm);


  //Update all elements
  if(rank > 0)
  {
    for ( auto& eachTuple : localVector) 
    {
      //Update Pn and Pc
      std::get<1>(eachTuple) = std::get<1>(eachTuple) + previousReadIdSum;
      std::get<2>(eachTuple) = std::get<1>(eachTuple);
    }
  }

  //For logging purpose, count total kmers across all the nodes
  auto localVecSize = localVector.size();

  //Global vector size
  uint64_t globalVecSize;
  MPI_Reduce(&localVecSize, &globalVecSize, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);

  if(rank == 0)
    std::cout << "Total count of tuples: " << globalVecSize << " \n";
}

#endif
