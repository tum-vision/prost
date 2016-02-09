find_package(Doxygen)

set(DOXYFILE ${CMAKE_CURRENT_SOURCE_DIR}/../Doxyfile)

add_custom_target(docs
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating documentation with Doxygen"
  VERBATIM)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ../docs)
