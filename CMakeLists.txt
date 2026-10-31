cmake_minimum_required(VERSION 3.16)
project(tinyllama_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd / VS Code IntelliSense

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TL_NATIVE    "Optimise for this machine's CPU (-march=native)" ON)
option(TL_OPENMP    "Multithreading with OpenMP" ON)
option(TL_SANITIZE  "Address + undefined-behaviour sanitizers (debug builds)" OFF)
option(TL_TINY_TEST "Toy model sizes for the smoke test" OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
if(TL_NATIVE)
  add_compile_options(-march=native)
endif()
if(TL_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_executable(tinyllama src/main.cpp src/model.cpp src/tokenizer.cpp src/kv_cache.cpp)
add_executable(bandwidth tools/bandwidth.cpp)

if(TL_TINY_TEST)
  target_compile_definitions(tinyllama PRIVATE TL_TINY_TEST)
endif()

if(TL_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tinyllama PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bandwidth PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found: building single-threaded (macOS: brew install libomp)")
  endif()
endif()
