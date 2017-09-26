# Sample mac clang-omp++ cmake call: cmake -DCMAKE_CXX_COMPILER=clang-omp++
# Regular call: cmake

cmake_minimum_required (VERSION 2.6)
set(CMAKE_CXX_COMPILER "/usr/bin/mpiCC")
project (MPIPLATFORM CXX)

# C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -g")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Ofast")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# gflags
add_subdirectory(gflags)

add_executable(mpiplatform src/main.cpp)
target_link_libraries(mpiplatform gflags armadillo)

