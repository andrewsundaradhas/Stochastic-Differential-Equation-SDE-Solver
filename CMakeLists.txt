cmake_minimum_required(VERSION 3.15)
project(sde-solver VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard and properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type settings
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX /O2 /MP)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)  # Disable min/max macros
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -march=native -pthread)
    
    # Different optimization levels based on build type
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG")
    
    # Enable OpenMP if available
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(${OpenMP_CXX_FLAGS})
    endif()
endif()

# Find required packages
find_package(Threads REQUIRED)

# Options
option(BUILD_TESTS "Build test cases" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# GoogleTest
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

# Library target
add_library(sde_solver
    src/models.cpp
)

target_include_directories(sde_solver 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(sde_solver PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(sde_solver PUBLIC cxx_std_17)
target_link_libraries(sde_solver PRIVATE Threads::Threads)

# Installation
include(GNUInstallDirs)
install(TARGETS sde_solver
    EXPORT sde-solver-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT sde-solver-targets
    FILE sde-solver-config.cmake
    NAMESPACE sde::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sde-solver
)

# Demo executables
if(BUILD_EXAMPLES)
    add_executable(vasicek_demo src/main_vasicek.cpp)
    target_link_libraries(vasicek_demo PRIVATE sde_solver)
    set_target_properties(vasicek_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS vasicek_demo DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(heston_demo src/main_heston.cpp)
    target_link_libraries(heston_demo PRIVATE sde_solver)
    set_target_properties(heston_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS heston_demo DESTINATION ${CMAKE_INSTALL_BINDIR})
    
    # Add examples to build
    add_custom_target(examples
        COMMAND ${CMAKE_COMMAND} -E echo "Examples built in ${CMAKE_BINARY_DIR}/bin"
        DEPENDS vasicek_demo heston_demo
    )
endif()

# Tests
if(BUILD_TESTS)
    # Vasicek model tests
    add_executable(test_vasicek tests/test_vasicek.cpp)
    target_link_libraries(test_vasicek PRIVATE sde_solver GTest::GTest GTest::Main)
    set_target_properties(test_vasicek PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    
    # Heston model tests
    add_executable(test_heston tests/test_heston.cpp)
    target_link_libraries(test_heston PRIVATE sde_solver GTest::GTest GTest::Main)
    set_target_properties(test_heston PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    
    enable_testing()
    add_test(NAME test_vasicek 
             COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1 $<TARGET_FILE:test_vasicek>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    
    add_test(NAME test_heston 
             COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1 $<TARGET_FILE:test_heston>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    
    # Add coverage if available
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
        if(ENABLE_COVERAGE)
            target_compile_options(test_vasicek PRIVATE --coverage -O0 -g)
            target_link_libraries(test_vasicek PRIVATE --coverage)
            
            target_compile_options(test_heston PRIVATE --coverage -O0 -g)
            target_link_libraries(test_heston PRIVATE --coverage)
        endif()
    endif()
    
    # Add a test target for convenience
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS test_vasicek test_heston
    )
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    
    # Python module
    pybind11_add_module(
        sde_solver
        python/bindings.cpp
    )
    
    target_include_directories(sde_solver PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(sde_solver PRIVATE sde_solver pybind11::module)
    
    # Set output directory
    set_target_properties(sde_solver PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    
    # Install Python module
    install(TARGETS sde_solver
        LIBRARY DESTINATION ${Python3_SITELIB}/sde_solver
        RUNTIME DESTINATION ${Python3_SITELIB}/sde_solver
    )
    
    # Install Python examples
    install(DIRECTORY examples/
        DESTINATION ${Python3_SITELIB}/sde_solver/examples
        FILES_MATCHING PATTERN "*.py"
    )
    
    # Create a virtual environment for development
    find_program(PYTHON3_EXECUTABLE python)
    if(PYTHON3_EXECUTABLE)
        add_custom_target(python_env
            COMMAND ${PYTHON3_EXECUTABLE} -m venv ${CMAKE_BINARY_DIR}/venv
            COMMAND ${CMAKE_COMMAND} -E echo "Python virtual environment created. Activate with:\n  source ${CMAKE_BINARY_DIR}/venv/bin/activate  # Linux/macOS\n  ${CMAKE_BINARY_DIR}\\venv\\Scripts\\activate  # Windows"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
    
    # Add a target to build and install the Python package
    add_custom_target(python_package
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
        COMMAND ${CMAKE_COMMAND} -E echo "Python package installed to ${Python3_SITELIB}"
        DEPENDS sde_solver
    )
    
    # Add a target to run the examples
    add_custom_target(run_examples
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/visualization.py
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS python_package
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sde-solver-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sde-solver-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sde-solver
)

# CPack configuration
set(CPACK_PACKAGE_VENDOR "SDE Solver Team")
set(CPACK_PACKAGE_CONTACT "your.email@example.com")
set(CPACK_PACKAGE_DESCRIPTION "A high-performance C++ SDE Solver framework")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.git/;/.vscode/;/.idea/;*.pyc;__pycache__/")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "sde-solver-${PROJECT_VERSION}")

# Add examples to source package
set(CPACK_SOURCE_INSTALLED_DIRECTORIES 
    "${CMAKE_CURRENT_SOURCE_DIR}/examples" "examples"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests" "tests"
)

include(CPack)

# Print configuration summary
message(STATUS "\n")
message(STATUS "SDE Solver Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
if(OpenMP_FOUND)
    message(STATUS "  OpenMP support: Enabled")
else()
    message(STATUS "  OpenMP support: Disabled")
endif()
message(STATUS "\nBuild targets:")
message(STATUS "  make all              - Build all targets")
if(BUILD_TESTS)
    message(STATUS "  make check            - Build and run tests")
endif()
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "  make python_package   - Build and install Python package")
    message(STATUS "  make run_examples     - Run Python examples")
endif()
message(STATUS "  make install          - Install the library")
message(STATUS "  make package          - Create source package")
message(STATUS "\n")
