cmake_minimum_required(VERSION 2.8)
project(ecto_example)

find_package(ecto REQUIRED)

#all the ecto modules will be put here.
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

find_package(OpenCV REQUIRED)

add_subdirectory(cells)

