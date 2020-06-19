# Author: petter.strandmark@gmail.com (Petter Strandmark)
CMAKE_MINIMUM_REQUIRED(VERSION 3.4.0)
PROJECT(MCTS CXX C)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
# Default locations to search for on various platforms.
if(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    LIST(APPEND SEARCH_LIBS /usr/lib)
    LIST(APPEND SEARCH_LIBS /usr/local/lib)
    LIST(APPEND SEARCH_LIBS /usr/local/opt/llvm/lib)

    LIST(APPEND SEARCH_HEADERS /usr/include)
    LIST(APPEND SEARCH_HEADERS /opt/local/include)
    LIST(APPEND SEARCH_HEADERS /usr/local/opt/llvm/include)
ENDIF(APPLE)

ENABLE_TESTING()

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ENDIF()
MESSAGE("-- The Build type is ${CMAKE_BUILD_TYPE}")
add_compile_options(-Wall -Wextra)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -ffast-math -march=native -funroll-loops -malign-double -minline-all-stringops -ftree-parallelize-loops=`grep -c ^processor /proc/cpuinfo
` -flto -fuse-linker-plugin ")
SET (MY_LIBRARY_DEPENDENCIES)

## OpenMP
FIND_PACKAGE(OpenMP REQUIRED)
IF(${OPENMP_FOUND})
    add_compile_options(${OpenMP_CXX_FLAGS})
    IF(APPLE)
        LIST(APPEND MY_LIBRARY_DEPENDENCIES omp)
    ELSE()
        LIST(APPEND MY_LIBRARY_DEPENDENCIES ${OpenMP_CXX_LIBRARIES})
    ENDIF()

# IF (APPLE)
  # OpenMP needs to be included as a library on some platforms.
  #  LIST(APPEND MY_LIBRARY_DEPENDENCIES omp)
# ENDIF (NOT MSVC)

ELSE(${OPENMP_FOUND})
MESSAGE("-- Can't find OpenMP. Continuing without it.")
ENDIF(${OPENMP_FOUND})

## Pytorch
FIND_PACKAGE(torch REQUIRED)
MESSAGE("-- Find PyTorch: ${TORCH_FOUND} ")
LIST(APPEND MY_LIBRARY_DEPENDENCIES ${TORCH_LIBRARIES})
LIST(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})

## Boost
SET(Boost_NO_BOOST_CMAKE ON)
FIND_PACKAGE(Boost COMPONENTS mpi serialization REQUIRED)
MESSAGE("-- The Boost version is ${Boost_VERSION}")
LIST(APPEND MY_LIBRARY_DEPENDENCIES ${Boost_LIBRARIES})
LIST(APPEND CMAKE_CXX_FLAGS ${Boost_CXX_FLAGS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

## MPI
FIND_PACKAGE(MPI COMPONENTS CXX C OPTIONAL_COMPONENTS FORTRAN)
LIST(APPEND CMAKE_CXX_FLAGS ${MPI_CXX_COMPILE_OPTIONS})
ADD_DEFINITIONS(${MPI_CXX_COMPILE_DEFINITIONS})
INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_DIRS})
LIST(APPEND MY_LIBRARY_DEPENDENCIES ${MPI_CXX_LIBRARIES})
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

# add spdlog
find_package(spdlog QUIET)
if(NOT ${spdlog_FOUND})
    message("-- NOT Found The spdlog try to compile ourself.")
    add_subdirectory(libspdlog)
endif()

# add spdlog
include(libck.cmake)

# Add marl
set(MARL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libmarl)
add_subdirectory(${MARL_DIR})

LIST(APPEND MY_LIBRARY_DEPENDENCIES spdlog::spdlog spdlog::spdlog_header_only ck marl)
#
# Include directories
#
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
FILE(GLOB MCTS_HEADERS ${CMAKE_SOURCE_DIR}/*.h)

ADD_SUBDIRECTORY(games)
