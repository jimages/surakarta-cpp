# Author: petter.strandmark@gmail.com (Petter Strandmark)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
PROJECT(MCTS CXX)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /usr/local/opt/llvm/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /opt/local/include)
LIST(APPEND SEARCH_HEADERS /usr/local/opt/llvm/include)

ENABLE_TESTING()

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.

IF(NOT CMAKE_BUILD_TYPE)
SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ENDIF()
MESSAGE("-- The Build type is ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
SET (MY_LIBRARY_DEPENDENCIES)

## OpenMP
FIND_PACKAGE(OpenMP)
IF(${OPENMP_FOUND})
MESSAGE("-- Found OpenMP.")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ADD_DEFINITIONS(-DUSE_OPENMP)

IF (NOT MSVC)
  # OpenMP needs to be included as a library on some platforms.
  LIST(APPEND MY_LIBRARY_DEPENDENCIES omp)
ENDIF (NOT MSVC)

ELSE(${OPENMP_FOUND})
MESSAGE("-- Can't find OpenMP. Continuing without it.")
ENDIF(${OPENMP_FOUND})

## Pytorch
FIND_PACKAGE(Torch REQUIRED)
LIST(APPEND MY_LIBRARY_DEPENDENCIES ${TORCH_LIBRARIES})

## Boost
FIND_PACKAGE(Boost REQUIRED
    COMPONENTS mpi)
LIST(APPEND MY_LIBRARY_DEPENDENCIES ${Boost_LIBRARIES})

## MPI
FIND_PACKAGE(MPI REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_OPTIONS}")
ADD_DEFINITIONS(${MPI_CXX_COMPILE_DEFINITIONS})
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
LIST(APPEND MY_LIBRARY_DEPENDENCIES ${MPI_CXX_LIBRARIES})
MESSAGE("-- ${MPI_CXX_LIBRARIES}")
#
IF (CMAKE_COMPILER_IS_GNUCXX)
  # No warnings for C. The Meschach library contains really old code.
  # -fPIC for building a shared library.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter -fpic")
ENDIF (CMAKE_COMPILER_IS_GNUCXX)

#
# Clang settings
#
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter -fpic")
ENDIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

#
# Include directories
#
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
FILE(GLOB MCTS_HEADERS ${CMAKE_SOURCE_DIR}/*.h)

ADD_SUBDIRECTORY(games)
