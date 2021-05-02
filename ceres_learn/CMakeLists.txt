cmake_minimum_required(VERSION 2.6)
project(ceres_learn)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package( Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS})
add_executable(ceres_learn  min.cpp)
target_link_libraries(ceres_learn ${CERES_LIBRARIES})

