

######## Compiler-specific options
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_HAS_C99_MATH /MP")   # -Wall produces 20k warnings
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
endif()


file(GLOB_RECURSE CURRENT_HEADERS  *.h)
file(GLOB CURRENT_SOURCES  *.cpp)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(eigen3 SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(eigen3 STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
set_target_properties(eigen3 PROPERTIES LINKER_LANGUAGE CXX)
