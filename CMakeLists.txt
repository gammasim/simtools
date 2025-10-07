cmake_minimum_required(VERSION 3.17...3.26 FATAL_ERROR)

project(simtools_lightemission LANGUAGES CXX)

# Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (prefer an existing install; fallback to FetchContent if missing)
include(FetchContent)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via find_package; fetching it...")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Allow using CMAKE_PREFIX_PATH or SIMTEL_PREFIX to locate the sim_telarray installation root.
# The observed layout (from user) is:
#   <root>/sim_telarray/sim_telarray/LightEmission/{IactLightEmission.hh, lib/libLightEmission.so}
# If SIMTEL_PREFIX points to <root>/sim_telarray/sim_telarray we add it; if it points higher
# we add additional candidate suffixes below.
if(DEFINED SIMTEL_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${SIMTEL_PREFIX}")
endif()

# Helper: broaden search to handle nested sim_telarray/sim_telarray structure.
set(_LE_PATH_SUFFIXES
  sim_telarray/sim_telarray/LightEmission/lib
  sim_telarray/LightEmission/lib
  LightEmission/lib
  sim_telarray/sim_telarray/LightEmission
  sim_telarray/LightEmission
  LightEmission
  lib
)
# Expect structure like:
#   <prefix>/sim_telarray/LightEmission/lib/libLightEmission.(so|dylib)
#   <prefix>/sim_telarray/LightEmission/IactLightEmission.hh
#   <prefix>/sim_telarray/corsika/iact.h
#   <prefix>/sim_telarray/common/rndm2.h

find_library(LightEmission_LIBRARY
  NAMES LightEmission
  PATH_SUFFIXES ${_LE_PATH_SUFFIXES}
  REQUIRED
)
message(STATUS "Found libLightEmission: ${LightEmission_LIBRARY}")

find_path(LightEmission_INCLUDE_DIR
  NAMES IactLightEmission.hh
  PATH_SUFFIXES ${_LE_PATH_SUFFIXES}
  REQUIRED
)
message(STATUS "Found LightEmission include: ${LightEmission_INCLUDE_DIR}")

set(_HESSIO_PATH_SUFFIXES
  hessioxxx/lib
  hessioxxx/lib64
  sim_telarray/hessioxxx/lib
  sim_telarray/hessioxxx/lib64
  hessio/lib
  hessio/lib64
  lib
)
set(_HESSIO_INC_SUFFIXES
  hessioxxx/include
  sim_telarray/hessioxxx/include
  hessio/include
  include
)

find_library(HESSIO_LIBRARY
  NAMES hessio
  PATH_SUFFIXES ${_HESSIO_PATH_SUFFIXES}
)
find_path(HESSIO_INCLUDE_DIR
  NAMES io_basic.h
  PATH_SUFFIXES ${_HESSIO_INC_SUFFIXES}
)

if(HESSIO_LIBRARY AND HESSIO_INCLUDE_DIR)
  message(STATUS "Found libhessio: ${HESSIO_LIBRARY}")
  message(STATUS "Found hessio include: ${HESSIO_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "hessio not found. Provide -DHESSIO_LIBRARY=/workdir/sim_telarray/hessioxxx/lib/libhessio.so -DHESSIO_INCLUDE_DIR=/workdir/sim_telarray/hessioxxx/include or set HESSIO_ROOT before re-running cmake.")
endif()

set(_IACT_PATH_SUFFIXES
  sim_telarray/sim_telarray/corsika
  sim_telarray/corsika
  corsika
  sim_telarray/sim_telarray/include
  sim_telarray/include
  include
)

if(NOT IACT_INCLUDE_DIR)
  find_path(IACT_INCLUDE_DIR
    NAMES iact.h
    PATH_SUFFIXES ${_IACT_PATH_SUFFIXES}
  )
endif()

if(NOT IACT_INCLUDE_DIR)
  message(FATAL_ERROR "Could not locate iact.h. Provide -DIACT_INCLUDE_DIR=/path/to/corsika (directory containing iact.h)")
else()
  message(STATUS "Found iact include: ${IACT_INCLUDE_DIR}")
endif()

set(_SIMTEL_COMMON_SUFFIXES
  sim_telarray/sim_telarray/common
  sim_telarray/common
  common
  sim_telarray/sim_telarray/include
  sim_telarray/include
  include
)
if(NOT SIMTEL_COMMON_INCLUDE_DIR)
  find_path(SIMTEL_COMMON_INCLUDE_DIR
    NAMES rndm2.h
    PATH_SUFFIXES ${_SIMTEL_COMMON_SUFFIXES}
  )
endif()
if(NOT SIMTEL_COMMON_INCLUDE_DIR)
  message(FATAL_ERROR "Could not locate rndm2.h. Provide -DSIMTEL_COMMON_INCLUDE_DIR=/path/with/rndm2.h")
else()
  message(STATUS "Found sim_telarray common include: ${SIMTEL_COMMON_INCLUDE_DIR}")
endif()

# Imported targets for third-party libs
add_library(LightEmission SHARED IMPORTED)
set_target_properties(LightEmission PROPERTIES IMPORTED_LOCATION ${LightEmission_LIBRARY})
target_include_directories(LightEmission INTERFACE ${LightEmission_INCLUDE_DIR} ${IACT_INCLUDE_DIR} ${SIMTEL_COMMON_INCLUDE_DIR})
target_link_libraries(LightEmission INTERFACE ${HESSIO_LIBRARY})

add_library(hessio SHARED IMPORTED)
set_target_properties(hessio PROPERTIES IMPORTED_LOCATION ${HESSIO_LIBRARY})
target_include_directories(hessio INTERFACE ${HESSIO_INCLUDE_DIR})

# Build the pybind11 module
set(BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/simtools/light_emission/bindings.cpp)
pybind11_add_module(_le MODULE ${BINDINGS_SRC})

target_include_directories(_le PRIVATE
  ${LightEmission_INCLUDE_DIR}
  ${IACT_INCLUDE_DIR}
  ${SIMTEL_COMMON_INCLUDE_DIR}
  ${HESSIO_INCLUDE_DIR}
)

target_link_libraries(_le PRIVATE LightEmission hessio)

# Some LightEmission builds expect these macros
target_compile_definitions(_le PRIVATE CTA_PROD3=1 IACT_NO_GRID=1)

# Place the built extension into the Python package folder so it can be imported in-place
set_target_properties(
  _le PROPERTIES
  OUTPUT_NAME "_le"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/simtools/light_emission
)

# Ensure runtime can locate LightEmission library
get_filename_component(LightEmission_LIB_DIR "${LightEmission_LIBRARY}" DIRECTORY)
get_filename_component(HESSIO_LIB_DIR "${HESSIO_LIBRARY}" DIRECTORY)

set(_RPATH_DIRS ${LightEmission_LIB_DIR} ${HESSIO_LIB_DIR})

# --- Minimal GSL detection (needed if libLightEmission not fully linked) ---
set(_GSL_HINTS "")
if(DEFINED ENV{GSL_ROOT})
  list(APPEND _GSL_HINTS $ENV{GSL_ROOT})
endif()
if(DEFINED GSL_ROOT)
  list(APPEND _GSL_HINTS ${GSL_ROOT})
endif()

find_library(GSL_LIBRARY NAMES gsl HINTS ${_GSL_HINTS} PATH_SUFFIXES lib lib64)
find_library(GSLCBLAS_LIBRARY NAMES gslcblas cblas HINTS ${_GSL_HINTS} PATH_SUFFIXES lib lib64)

if(GSL_LIBRARY)
  message(STATUS "Linking with GSL: ${GSL_LIBRARY}")
  target_link_libraries(_le PRIVATE ${GSL_LIBRARY})
  get_filename_component(_GSL_LIB_DIR "${GSL_LIBRARY}" DIRECTORY)
  list(APPEND _RPATH_DIRS ${_GSL_LIB_DIR})
else()
  message(WARNING "GSL not found (gsl_rng_ranlux unresolved?): set GSL_ROOT or -DGSL_ROOT to fix.")
endif()
if(GSLCBLAS_LIBRARY)
  message(STATUS "Linking with GSLCBLAS: ${GSLCBLAS_LIBRARY}")
  target_link_libraries(_le PRIVATE ${GSLCBLAS_LIBRARY})
  get_filename_component(_GSLCBLAS_LIB_DIR "${GSLCBLAS_LIBRARY}" DIRECTORY)
  list(APPEND _RPATH_DIRS ${_GSLCBLAS_LIB_DIR})
endif()
list(REMOVE_DUPLICATES _RPATH_DIRS)
string(REPLACE ";" ":" _RPATH_STR "${_RPATH_DIRS}")

set_target_properties(_le PROPERTIES BUILD_WITH_INSTALL_RPATH ON)
set_target_properties(_le PROPERTIES INSTALL_RPATH "${_RPATH_STR}" BUILD_RPATH "${_RPATH_STR}")
