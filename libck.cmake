#libck
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
set(LIBCK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libck)
set(LIBCK_BIN ${CMAKE_CURRENT_BINARY_DIR}/libck)
set(LIBCK_STATIC_LIB ${LIBCK_BIN}/lib/libck.a)
set(LIBCK_INCLUDES ${LIBCK_BIN}/include)
file(MAKE_DIRECTORY ${LIBCK_INCLUDES})

ExternalProject_Add(
    libck
    PREFIX ${LIBCK_BIN}
    SOURCE_DIR ${LIBCK_DIR}
    CONFIGURE_COMMAND ${LIBCK_DIR}/configure --prefix=${LIBCK_BIN}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${LIBCK_STATIC_LIB}
)
add_library(ck STATIC IMPORTED GLOBAL)

add_dependencies(ck libck)
set_target_properties(ck PROPERTIES IMPORTED_LOCATION ${LIBCK_STATIC_LIB})
set_target_properties(ck PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${LIBCK_INCLUDES})
