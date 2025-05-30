cmake_minimum_required(VERSION 3.24)
project(SimulacionCalor C)

set(CMAKE_C_STANDARD 11)

# Añadir flag para OpenMP
find_package(OpenMP REQUIRED)

# Crear el ejecutable antes de usar target_link_libraries
add_executable(SimulacionCalor main.c)

if(OpenMP_C_FOUND)
    target_link_libraries(SimulacionCalor PUBLIC OpenMP::OpenMP_C)
endif()

