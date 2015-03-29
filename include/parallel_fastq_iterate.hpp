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



/**
 * Generate a vector of kmers from FASTQ file for each MPI process
 * This vector will no doubt have duplicate entries
 * Approach to build the vector
 * 1. Define file blocks and iterators for each rank
 * 2. Within each rank, iterate over all the reads
 * 3. For each read, iterate over all the kmers and push them to vector
 * 4. Return the vector
 *
 * @tparam T                Type of vector to populate
 * @param[out] localVector  Reference of vector to populate
 * @note                    This function should be called by all MPI ranks
 */
template <typename KmerType, typename Alphabet, typename T>
void generateKmerVector(MPI_Comm comm, const std::string &filename,
                        T& localVector) 
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
        //Insert kmer to vector
        localVector.push_back(*start);
      }
    }

    partition = loader.getNextL1Block();
  }

  std::cerr << "[generateKmerVector] Rank : " << rank << " parsed " << localVector.size() << " kmers\n";
}
