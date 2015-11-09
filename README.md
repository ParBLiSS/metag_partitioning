# Parallel metegenome assembler for large soil datasets #

### Authors ###

* Chirag Jain
* Patrick Flick
* Tony Pan

### Introduction ###

Parallel metagenomic assembler designed to handle very large datasets. Program identifies the disconnected subgraphs in the de Bruijn graph, partitions the input dataset and runs a popular assember **Velvet** independently on the partitions. This software is a high performance version of the [khmer](http://khmer.readthedocs.org/en/v2.0/) library for assembly. 

The whole algorithm relies on the existence of disconnected components in the de Bruijn graph for the performance gains. We found that this assumption is generally true for the currently available soil datasets from forests and agriculture land.

### Install ###


The repository and external submodules can be cloned directly:

    git clone --recursive <GITHUB_URL>
    mkdir build_directory
    cd build_directory
    cmake ../Metag_partitioning
    make
    make

### Run ###

Inside the build directory, 

    mpirun -np <COUNT OF PROCESSES> ./bin/metaG --file <FASTQ_FILE> --velvetK <KMER_SIZE_FOR_ASSEMBLY>
    Eg. mpirun -np 8 ./bin/metaG --file sample.fastq --velvetK 45

We have some sample files in the data folder of the code, you can use those for trial runs. You should see a file called contigs.fa containing all the assembled contigs after the run is successful. 

### Customization (required) ###

During the assembly, velvet does file I/O to save intermediate results. Therefore you need to specify the paths suitable for it. Please check the files include/config files . These 2 files contain all the parameters that can be tuned by the users. 

### Dependency ###

It requires C++ 11 features (gcc 4.7 or above) and MPI

External git submodules (automatically downloaded and compiled):

* [BLISS](https://bitbucket.org/AluruLab/bliss)
* [mxx](https://github.com/patflick/mxx)
* [velvet](https://github.com/dzerbino/velvet)
* [R-MAT generator](http://www.graph500.org/)

### Cite ###

Please cite the following publication if you are using this code for your research:

A Parallel Connectivity Algorithm for de-Bruijn Graphs in Metagenomic Applications. Patrick Flick, Chirag Jain, Tony Pan, and Srinivas Aluru. Proceedings of 2015 International Conference for High Performance Computing, Networking, Storage and Analysis. ACM, 2015.
