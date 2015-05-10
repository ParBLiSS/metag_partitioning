#ifndef PACKED_READ_H
#define PACKED_READ_H

//Includes from BLISS
#include <common/alphabets.hpp>
#include <common/alphabet_traits.hpp>
#include <common/padding.hpp>


//Container of storage information of the reads
//Required to evaluate the size of std::array for keeping read sequence
template <typename ALPHABET, typename WORD_TYPE>
struct readStorageInfo
{
    //Maximum read count in terms of nuc characters
    static constexpr unsigned int maxCharCount = MAX_READ_SIZE;

    // The number of bits of each character
    static constexpr unsigned int bitsPerChar = bliss::common::AlphabetTraits<ALPHABET>::getBitsPerChar();

    // The total number of bits
    static constexpr unsigned int nBits = maxCharCount * bitsPerChar;

    //Need to determine amount of storage for keeping nBits bits inside the word_type array 
    typedef typename bliss::common::UnpaddedStreamTraits<WORD_TYPE, nBits> bitstream;
    static constexpr unsigned int nWords = bitstream::nWords;

    typedef ALPHABET ReadAlphabet;
    typedef WORD_TYPE ReadWordType;
};

//Helper function for packing read function
//Places the alphabet at an appropriate position in the contiguous array
template<typename ReadInf, typename T, std::size_t N, typename WType>
inline void setBitsAtPos(std::array<T, N>& readPacked, WType w, int bitPos, int numBits) 
{
  
  //Storage info about packing read
  typedef typename ReadInf::ReadWordType wordtype;
  int bitsPerWord = ReadInf::bitstream::bitsPerWord;
  
  // get the value to insert masked first
  wordtype charVal = static_cast<wordtype>(w) & getLeastSignificantBitsMask<wordtype>(numBits);

  // determine which word in the array it needs to go to
  int wordId = bitPos / bitsPerWord;
  int offsetInWord = bitPos % bitsPerWord;  // offset is where the LSB of the char will sit, in bit coordinate.

  if (offsetInWord >= (bitsPerWord - numBits)) {
    // if split between words, deal with it.
    readPacked[wordId] |= charVal << offsetInWord;   // the lower bits of the charVal

    // the number of lowerbits consumed is (bitstream::bitsPerWord - offsetInWord)
    // so right shift those many places and what remains goes into the next word.
    if (wordId < ReadInf::nWords - 1) readPacked[wordId + 1] |= charVal >> (bitsPerWord - offsetInWord);


  } else {
    // else insert into the specific word.
    readPacked[wordId] |= charVal << offsetInWord;
  }
}

/*
 * @brief     returns read characters packed as bits inside a vector
 * @param[in] start             Beginning of the sequence iterator in a read
 * @param[in] end               Ending
 *
 * Note that InputIterator has value domain consistent with the valid values in the alphabet
 *
 * @param[out] readPacked       Read content stored inside this vector
 * @param[out] readCharCount    Count of characters present in the given read
 */
template <typename InputIterator, typename ReadInf, typename T, std::size_t N>
void getPackedRead(std::array<T, N>& readPacked, uint32_t& readCharCount, InputIterator start, InputIterator end)
{
  //Assign the value to readPackedSize
  readCharCount = end - start;

  //Make sure readPackedSize is below 
  assert(readCharCount <= ReadInf::maxCharCount);

  int bitPos = ReadInf::bitstream::nBits - ReadInf::bitsPerChar;

  for (auto iter = start; iter!=end ; bitPos -= ReadInf::bitsPerChar) 
  {
    //Insert the alphabet in the array
    setBitsAtPos<ReadInf>(*iter, bitPos, ReadInf::bitsPerChar);
  }
}

#endif
