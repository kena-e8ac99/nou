cmake_minimum_required(VERSION 3.28)

project(nou LANGUAGES CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

add_library(nou)
target_sources(nou PUBLIC FILE_SET CXX_MODULES FILES include/nou/nou.cppm)

if(PROJECT_IS_TOP_LEVEL)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wshadow)
  elseif("MSVC")
    add_compile_options(/W4 /WX)
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

