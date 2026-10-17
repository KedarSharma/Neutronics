cmake_minimum_required(VERSION 3.18)

project(neutron_transport LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(neutron_simulation STATIC neutron_simulation.cu)
target_include_directories(neutron_simulation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET neutron_simulation PROPERTY CUDA_ARCHITECTURES 89)

add_executable(u238_monte_carlo u238_monte_carlo.cu)
set_property(TARGET u238_monte_carlo PROPERTY CUDA_ARCHITECTURES 89)

add_executable(natural_uranium_monte_carlo natural_uranium_monte_carlo.cu)
set_property(TARGET natural_uranium_monte_carlo PROPERTY CUDA_ARCHITECTURES 89)

add_executable(h2o_uranium_monte_carlo h2o_uranium_monte_carlo.cu)
set_property(TARGET h2o_uranium_monte_carlo PROPERTY CUDA_ARCHITECTURES 89)

add_executable(d2o_natural_uranium_monte_carlo d2o_natural_uranium_monte_carlo.cu)
set_property(TARGET d2o_natural_uranium_monte_carlo PROPERTY CUDA_ARCHITECTURES 89)
