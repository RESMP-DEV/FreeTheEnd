cmake_minimum_required(VERSION 3.20)
project(mc189 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(MC189_BUILD_TESTS "Build tests" ON)
option(MC189_ENABLE_VALIDATION "Enable Vulkan validation layers by default" OFF)
option(MC189_USE_MOLTENVK "Use MoltenVK on macOS" ON)
option(CPU_ONLY "Build with CPU backend only (no Vulkan dependency)" OFF)

if(NOT CPU_ONLY)
    # Vulkan SDK
    find_package(Vulkan REQUIRED)

    if(NOT Vulkan_FOUND)
        message(FATAL_ERROR "Vulkan SDK not found. Install via: brew install vulkan-sdk (macOS) or apt install vulkan-sdk (Linux)")
    endif()

    message(STATUS "Vulkan found: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan libraries: ${Vulkan_LIBRARIES}")

    # Platform-specific setup
    if(APPLE)
        message(STATUS "Building for macOS with MoltenVK support")

        # MoltenVK is included in the Vulkan SDK on macOS
        # Check for MoltenVK library
        if(MC189_USE_MOLTENVK)
            find_library(MOLTENVK_LIBRARY MoltenVK PATHS
                $ENV{VULKAN_SDK}/lib
                /usr/local/lib
                /opt/homebrew/lib
            )
            if(MOLTENVK_LIBRARY)
                message(STATUS "MoltenVK found: ${MOLTENVK_LIBRARY}")
            else()
                message(WARNING "MoltenVK not found, using default Vulkan loader")
            endif()
        endif()

        # macOS frameworks
        find_library(COCOA_LIBRARY Cocoa REQUIRED)
        find_library(IOKIT_LIBRARY IOKit REQUIRED)
        find_library(METAL_LIBRARY Metal REQUIRED)
        find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)

        set(PLATFORM_LIBS
            ${COCOA_LIBRARY}
            ${IOKIT_LIBRARY}
            ${METAL_LIBRARY}
            ${QUARTZCORE_LIBRARY}
        )

        # Deployment target
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target")

    elseif(WIN32)
        message(STATUS "Building for Windows")
        set(PLATFORM_LIBS "")

    elseif(UNIX)
        message(STATUS "Building for Linux")
        set(PLATFORM_LIBS "")
    endif()

    # mc189 library (Vulkan compute backend)
    add_library(mc189 STATIC
        src/vulkan_context.cpp
        src/compute_pipeline.cpp
        src/buffer_manager.cpp
        src/fortress_location.cpp
        src/spawn_selection.cpp
        src/world_seed_impl.cpp
    )

    target_include_directories(mc189 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${Vulkan_INCLUDE_DIRS}
    )

    target_link_libraries(mc189 PUBLIC
        ${Vulkan_LIBRARIES}
        ${PLATFORM_LIBS}
    )

    # Compiler warnings (relaxed for Vulkan SDK compatibility)
    if(MSVC)
        target_compile_options(mc189 PRIVATE /W4)
    else()
        target_compile_options(mc189 PRIVATE
            -Wall -Wextra -Wpedantic
            -Wno-missing-field-initializers  # vulkan.hpp triggers this
            -Wno-deprecated-declarations     # newer Vulkan SDK deprecations
            -Wno-unused-parameter
        )
    endif()

    # Vulkan.hpp configuration
    target_compile_definitions(mc189 PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_CONSTRUCTORS=0
        VK_ENABLE_BETA_EXTENSIONS=1
    )

    if(APPLE)
        target_compile_definitions(mc189 PUBLIC
            VK_USE_PLATFORM_METAL_EXT=1
        )
    elseif(WIN32)
        target_compile_definitions(mc189 PUBLIC
            VK_USE_PLATFORM_WIN32_KHR=1
        )
    elseif(UNIX)
        target_compile_definitions(mc189 PUBLIC
            VK_USE_PLATFORM_XCB_KHR=1
        )
    endif()

    if(MC189_ENABLE_VALIDATION)
        target_compile_definitions(mc189 PUBLIC MC189_ENABLE_VALIDATION=1)
    endif()
else()
    message(STATUS "CPU_ONLY mode: skipping Vulkan dependency")
endif()

# mc189_core library (game simulation)
if(CPU_ONLY)
    # On macOS with Apple Clang, OpenMP requires libomp from Homebrew
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE BREW_RESULT
        )
        if(BREW_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        else()
            message(STATUS "libomp not found via Homebrew, install with: brew install libomp")
        endif()
    endif()

    find_package(OpenMP REQUIRED)

    # Static library for CPU-only builds (only needed for tests)
    if(MC189_BUILD_TESTS)
        add_library(mc189_core_static STATIC
            src/cpu_backend.cpp
            src/world_seed_impl.cpp
            src/terrain_generator.cpp
            src/entity_manager.cpp
            src/mob_ai.cpp
            src/inventory.cpp
            src/crafting.cpp
            src/portal_manager.cpp
            src/observation_encoder.cpp
            src/reward_computer.cpp
            src/dimension.cpp
        )

        target_include_directories(mc189_core_static PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )

        target_link_libraries(mc189_core_static PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(mc189_core_static PUBLIC MC189_CPU_ONLY=1)
    endif()
else()
    # Static library for GPU builds (only needed for tests)
    if(MC189_BUILD_TESTS)
        add_library(mc189_core_static STATIC
            src/mc189_simulator.cpp
            src/multistage_simulator.cpp
            src/terrain_generator.cpp
            src/entity_manager.cpp
            src/mob_ai.cpp
            src/inventory.cpp
            src/crafting.cpp
            src/portal_manager.cpp
            src/observation_encoder.cpp
            src/reward_computer.cpp
            src/dimension.cpp
        )

        target_include_directories(mc189_core_static PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )

        target_link_libraries(mc189_core_static PUBLIC mc189)
    endif()
endif()

if(MC189_BUILD_TESTS AND MSVC)
    target_compile_options(mc189_core_static PRIVATE /W4)
elseif(MC189_BUILD_TESTS)
    target_compile_options(mc189_core_static PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
endif()

if(NOT CPU_ONLY)
    # Shader compilation
    find_program(GLSLC glslc HINTS /opt/homebrew/bin $ENV{VULKAN_SDK}/bin)

    if(GLSLC)
        file(GLOB COMPUTE_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
        set(SPV_FILES "")

        foreach(SHADER ${COMPUTE_SHADERS})
            get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
            set(SPV_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.spv")
            list(APPEND SPV_FILES ${SPV_FILE})

            add_custom_command(
                OUTPUT ${SPV_FILE}
                COMMAND ${GLSLC} -fshader-stage=compute ${SHADER} -o ${SPV_FILE}
                DEPENDS ${SHADER}
                COMMENT "Compiling ${SHADER_NAME}.comp"
            )
        endforeach()

        add_custom_target(compile_shaders DEPENDS ${SPV_FILES})
        message(STATUS "glslc found: ${GLSLC} - shader compilation enabled")
    else()
        message(WARNING "glslc not found, shaders will not be compiled")
    endif()

    # Example executable (requires Vulkan)
    add_executable(mc189_example examples/compute_example.cpp)
    target_link_libraries(mc189_example PRIVATE mc189)
endif()

# Test executable
if(MC189_BUILD_TESTS)
    enable_testing()

    # Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    if(NOT CPU_ONLY)
        # Simple Vulkan test without external framework
        add_executable(mc189_vulkan_test tests/test_vulkan.cpp)
        target_link_libraries(mc189_vulkan_test PRIVATE mc189)
        add_test(NAME vulkan_test COMMAND mc189_vulkan_test)

        # Vulkan backend integration tests
        add_executable(mc189_test
            tests/test_backend_integration.cpp
        )
        target_link_libraries(mc189_test PRIVATE
            mc189
            GTest::gtest_main
        )
        target_include_directories(mc189_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        gtest_discover_tests(mc189_test)

        # Multistage test (depends on full GPU pipeline)
        add_executable(mc189_multistage_test
            tests/test_multistage.cpp
        )
        target_link_libraries(mc189_multistage_test PRIVATE
            mc189_core
            GTest::gtest_main
        )
        target_include_directories(mc189_multistage_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        gtest_discover_tests(mc189_multistage_test)
    endif()

    include(GoogleTest)
endif()

# Install
if(NOT CPU_ONLY)
    install(TARGETS mc189
        EXPORT mc189Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    install(EXPORT mc189Targets
        FILE mc189Config.cmake
        NAMESPACE mc189::
        DESTINATION lib/cmake/mc189
    )
endif()

if(MC189_BUILD_TESTS)
    install(TARGETS mc189_core_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
endif()

install(DIRECTORY include/mc189 DESTINATION include)

# pkg-config
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/mc189.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/mc189.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mc189.pc DESTINATION lib/pkgconfig)

# Python bindings (optional)
include(cmake/pybind11.cmake)

# Print configuration summary
message(STATUS "")
message(STATUS "mc189 configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  CPU only: ${CPU_ONLY}")
if(NOT CPU_ONLY)
    message(STATUS "  Validation: ${MC189_ENABLE_VALIDATION}")
endif()
message(STATUS "  Build tests: ${MC189_BUILD_TESTS}")
message(STATUS "")
