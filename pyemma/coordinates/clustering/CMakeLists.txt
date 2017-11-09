cmake_minimum_required (VERSION 2.8.12)
project(pyemma-clustering)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)

pybind11_add_module(kmeans src/clustering_module.cpp)

# add mdtraj c-api.
find_package(PythonInterp)
if (! ${PYTHON_FOUND})
    message(FATAL_ERROR no python)
endif()

EXECUTE_PROCESS(COMMAND "${PYTHON_EXECUTABLE}" -c "import mdtraj; print(mdtraj.capi()['include_dir'])"
        OUTPUT_VARIABLE MDTRAJ_INCLUDE)

if (MDTRAJ_INCLUDE STREQUAL "")
    message(FATAL_ERROR "mdtraj lib not found")
endif()

EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import mdtraj; print(mdtraj.capi()['lib_dir'])"
        OUTPUT_VARIABLE MDTRAJ_LIB)

include_directories(include
                    ${MDTRAJ_INCLUDE})
link_directories(${MDTRAJ_LIB})
target_link_libraries(kmeans PUBLIC theobald)
