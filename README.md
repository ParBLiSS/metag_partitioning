# Parallel metegenome assembler #

### Authors ###

* Chirag Jain
* Patrick Flick
* Tony Pan

### Introduction ###

Parallel metagenomic assembler designed to handle very large datasets. Program identifies the disconnected subgraphs in the de Bruijn graph, partitions the input dataset and runs a popular assember **Velvet** independently on the partitions. This software is a high performance version of the [khmer](khmer.readthedocs.org) library for assembly.

### Install ###


The repository and external submodules can be cloned directly:

    git clone --recursive <GITHUB_URL>
    mkdir build_directory
    cd build_directory
    cmake ../Metag_partitioning
    make 

### Run ###

Inside the build directory, 

    mpirun -np <COUNT OF PROCESSES> ./bin/getHistogram --file <FASTQ_FILE> --velvetK <KMER_SIZE_FOR_ASSEMBLY>
    Eg. mpirun -np 8 ./bin/getHistogram --file sample.fastq --velvetK 45

### Customize ###

Please check the files include/config* . These 2 files contain all the parameters that can be tuned by the users. 

### Dependency ###

gcc 4.7 or above for C++11
External git submodules (automatically downloaded and compiled):

* BLISS
* mxx
* velvet