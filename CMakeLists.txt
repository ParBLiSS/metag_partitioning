cmake_minimum_required(VERSION 2.6)

# project settings
project(Metagenomics)

##### General Compilation Settings


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized --std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -msse3 -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
    set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()


#### MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
    #set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)


###### Executable and Libraries
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory("${CMAKE_SOURCE_DIR}/ext/bliss")
#add_subdirectory("${CMAKE_SOURCE_DIR}/ext/psac-copy")

#include_directories("${bliss_INCLUDE_DIRS}")
# TODO: eventually make BLISS define its own include dirs
include_directories("${bliss_SOURCE_DIR}/src")
include_directories("${bliss_BINARY_DIR}")

#include_directories("${CMAKE_SOURCE_DIR}/ext/psac-copy/include")
include_directories("${CMAKE_SOURCE_DIR}/ext")
include_directories("${CMAKE_SOURCE_DIR}/ext/graph500")
#include_directories("/home/tpan/src/bfs_debruijn")

# add own subdirectories
add_subdirectory("${CMAKE_SOURCE_DIR}/src")

### Velvet
message(STATUS "Compiling velvet")
set (COMPILE_VELVET make MAXKMERLENGTH=63)
execute_process(
  COMMAND ${COMPILE_VELVET}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/ext/velvet"
  OUTPUT_QUIET ERROR_QUIET)

### Graph500 generator
message(STATUS "Compiling Graph500 generator")
set (COMPILE_GRAPH500_GEN make -f Makefile.mpi all)
execute_process(
	COMMAND ${COMPILE_GRAPH500_GEN}
	WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/ext/graph500/generator"
  OUTPUT_QUIET ERROR_QUIET)
	

