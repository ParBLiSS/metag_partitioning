//Includes
#include <iostream>
#include <map>
#include <string>
#include <fstream>

//File includes from BLISS

//Own includes

template <typename valueType, typename keyType>
void createPartitionKmerMap(std::string filename, std::multimap<keyType, valueType> &pidKmersMap)
{
  std::ifstream infile(filename);
  keyType p_id;
  valueType kmer;

  //Read the file
  while(infile >> kmer >> p_id)
  {
    //Insert to map
    pidKmersMap.insert(std::make_pair(p_id, kmer));
  }

  int countPartitions = 0;
  for( auto iter = pidKmersMap.begin() ; iter != pidKmersMap.end() ; iter = pidKmersMap.upper_bound( iter->first )  )
      ++countPartitions;

  std::cout << filename << " has " << countPartitions << " partitions.\n";
}


int main(int argc, char** argv)
{
  //Specify the fileName
  std::string filename1; 
  std::string filename2; 

  bool compareTwoFile;

  if( argc == 3 ) {
    filename1 = argv[1];
    filename2 = argv[2];
    compareTwoFile = true;
  }
  else if( argc == 2 ) {
    filename1 = argv[1];
    compareTwoFile = false;
  }
  else {
    std::cout << "Usage : \n";
    std::cout << "<executable> <outputFile1>\n";
    std::cout << "<executable> <outputFile1> <outputFile2> \n";
    return 1;
  }

  //Assuming kmer-length is less than 32
  typedef uint64_t KmerIdType;

  //Assuming partition count is less than 4 Billion
  typedef uint32_t partitionIdType;

  std::multimap<partitionIdType, KmerIdType> pid_KmersMap1;
  std::multimap<partitionIdType, KmerIdType> pid_KmersMap2;

  //Create multimap from output files
  createPartitionKmerMap(filename1, pid_KmersMap1);
  if(compareTwoFile)
    createPartitionKmerMap(filename2, pid_KmersMap2);

  //Compare two maps if asked by user
  if(compareTwoFile)
  {
    bool resultsAreSame = (pid_KmersMap1 == pid_KmersMap2);

    if (resultsAreSame)
      std::cout << "SUCCESS : Contents in both file matches\n";
    else
      std::cout << "FAILURE : Contents in both file doesn't match\n";
  }

  return(0);
}

