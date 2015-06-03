#ifndef CONFIG_HPP
#define CONFIG_HPP

/*
 * ADJUSTABLE BY USER
 */

//Histogram equalization threshold
constexpr int HIST_EQ_THRESHOLD = 20; 

//Controls the threshold for read filtering
constexpr int KMER_FREQ_THRESHOLD = 50;

//Kmer length during filtering phase
constexpr int KMER_LEN_PRE = 20;

//Kmer length during de Bruijn graph partitioning
//For now, keep it <= 32
constexpr int KMER_LEN = 31;

//Kmer length for assembly
constexpr int VELVET_KMER_LEN = 45;

//We will discard partitions with less than these many reads
constexpr int MIN_READ_COUNT_FOR_ASSEMBLY = 5;

//This parameter should be set to greater than or equal to 
//the maximum read size expected in the dataset
const unsigned int MAX_READ_SIZE=128;





/*
 * NOT SUPPOSED TO BE CHANGED BY USER
 */

//Assuming read count is less than 4 Billion
//NOTE: Not sure about the correctness at the moment if following type is changed
typedef uint32_t ReadIdType;

//Type definition for partition id
typedef uint32_t PidType;

const unsigned int MAX = std::numeric_limits<ReadIdType>::max();
const unsigned int MAX_INT = std::numeric_limits<int>::max();

//Order of layers in kmer tuples (kmer, Pn, Pc)
class kmerTuple {
  public:
    static const uint8_t kmer = 0, Pn = 1, Pc =2;
};

//Order of layers in read sequence tuples (Sequence, readid, partitionid, count of nuc. characters in the read)
class readTuple {
  public:
    static const uint8_t seq = 0, rid = 1, pid = 2, cnt = 3;
};


#endif
