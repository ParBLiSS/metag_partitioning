#ifndef CONFIG_HPP
#define CONFIG_HPP

/*
 * ADJUSTABLE BY USER
 */

//Histogram equalization threshold
//Can be modified
constexpr int HIST_EQ_THRESHOLD = 10; 

//Controls the threshold for read filtering
//Can be modified
constexpr int KMER_FREQ_THRESHOLD = 50;

//Kmer length during filtering phase
//Keep it <= 32
constexpr int KMER_LEN_PRE = 20;

//Kmer length during de Bruijn graph partitioning
//Keep it <= 32
constexpr int KMER_LEN = 31;

//We will discard partitions with less than these many reads
//Can be modified
constexpr int MIN_READ_COUNT_FOR_ASSEMBLY = 5;

//This parameter should be set to greater than or equal to 
//the maximum read size expected in the dataset
//Can be modified
const unsigned int MAX_READ_SIZE=128;

//Print some more log output
#define DEBUGLOG 0

/*
 * NOT SUPPOSED TO BE CHANGED BY USER
 */

//Assuming kmer-length is less than 32
typedef uint64_t KmerIdType;

//Assuming read count is less than 4 Billion
//NOTE: Not sure about the correctness at the moment if following type is changed
typedef uint32_t ReadIdType;

//Type definition for partition id
typedef ReadIdType PidType;

//Type for defining read length
typedef uint16_t ReadLenType;

//Types for frequency and serialNos used during filtering phase 
typedef uint16_t KmerFreqType;
typedef uint16_t KmerSNoType;

const unsigned int MAX = std::numeric_limits<ReadIdType>::max();
const unsigned int MAX_INT = std::numeric_limits<int>::max();
const uint16_t MAX_FREQ = std::numeric_limits<KmerFreqType>::max();

//Order of layers in kmer tuples (kmer, Pn, Pc)
class kmerTuple {
  public:
    static const uint8_t kmer = 0, Pn = 1, Pc =2;
};

//Order of layers in kmer tuples during filtering phase (kmer, readId, freq, kmer_serial_no)
class kmerTuple_Pre {
  public:
    static const uint8_t kmer = 0, rid = 1, freq =2, kmer_sno=3;
};

//Order of layers in read sequence tuples (Sequence, readid, partitionid, count of nuc. characters in the read)
class readTuple {
  public:
    static const uint8_t seq = 0, rid = 1, pid = 2, cnt = 3;
};

//Struct to save command line options
struct cmdLineParams {
  //Fastq file containing the reads
  std::string fileName;

  //Kmer size to run Velvet
  int velvetKmerSize;

  //Name of the method (Only used for log-sort)
  std::string method;
};


//Struct to save command line options
struct cmdLineParamsGraph500 {
  //scale of the graph = log(num of vertices)
  size_t scale;

  //average edge degree of vertex in the graph.
  size_t edgefactor;

  //Kmer size to run Velvet
  int velvetKmerSize;

  //Name of the method (Only used for log-sort)
  std::string method;
};

/*
 * MXX TIMER
 */
#define MP_ENABLE_TIMER 1
#if MP_ENABLE_TIMER
#define MP_TIMER_START() mxx::section_timer timer;
#define MP_TIMER_END_SECTION(str) timer.end_section(str);
#else
#define MP_TIMER_START()
#define MP_TIMER_END_SECTION(str)
#endif

#endif
