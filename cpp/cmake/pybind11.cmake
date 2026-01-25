# CMake configuration for Python bindings

# Option to enable Python bindings
option(MC189_BUILD_PYTHON "Build Python bindings" ON)

if(MC189_BUILD_PYTHON)
    include(GNUInstallDirs)

    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)

    if(Python3_FOUND)
        set(_mc189_default_site_packages "${Python3_SITEARCH}")
        if(NOT _mc189_default_site_packages)
            set(_mc189_default_site_packages "${Python3_SITELIB}")
        endif()
    endif()

    if(NOT _mc189_default_site_packages)
        set(_mc189_default_site_packages "${CMAKE_INSTALL_LIBDIR}/python")
    endif()

    set(MC189_PYTHON_INSTALL_DIR
        "${_mc189_default_site_packages}"
        CACHE PATH "Install location for mc189 Python module"
    )

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Fetch pybind11
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    # Create Python module
    pybind11_add_module(mc189_core
        src/bindings/mc189_module.cpp
        src/bindings/py_vulkan_context.cpp
        src/bindings/py_buffer_manager.cpp
        src/bindings/py_compute_pipeline.cpp
        src/bindings/py_simulator.cpp
        src/mc189_simulator.cpp
    )

    target_link_libraries(mc189_core PRIVATE mc189)

    if(APPLE)
        set(_mc189_rpath "@loader_path")
    elseif(UNIX)
        set(_mc189_rpath "\$ORIGIN")
    endif()

    if(_mc189_rpath)
        set_target_properties(mc189_core PROPERTIES
            BUILD_WITH_INSTALL_RPATH YES
            INSTALL_RPATH "${_mc189_rpath}"
        )
    endif()

    install(TARGETS mc189_core
        LIBRARY DESTINATION "${MC189_PYTHON_INSTALL_DIR}"
        RUNTIME DESTINATION "${MC189_PYTHON_INSTALL_DIR}"
        ARCHIVE DESTINATION "${MC189_PYTHON_INSTALL_DIR}"
    )
endif()
