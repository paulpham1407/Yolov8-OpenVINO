cmake_minimum_required(VERSION 3.12)
set(PROJECT_NAME libinferenceopv)
project(${PROJECT_NAME})

# Linker external library
if(MSVC)
    if(NOT OpenCV_DIR)
        set(OpenCV_DIR "C:/opencv")
    endif()
endif()

find_package(OpenCV REQUIRED)

find_package(OpenCV REQUIRED)

if(MSVC)
    set(OPENVINO_DIR "C:/OpenVINO/runtime/")
else()
    set(OPENVINO_DIR "/opt/intel/openvino/runtime")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OPENVINO_DIR}/include
    ${OPENVINO_DIR}/include/ie
    ${OPENVINO_DIR}/include/ngraph
)

message(STATUS "OpenCV library status:")
message(STATUS "version:${OpenCV_VERSION}")
message(STATUS "libraries:${OpenCV_LIBS}")
message(STATUS "include path:${OpenCV_INCLUDE_DIRS}")

### generate detector.so
add_library(${PROJECT_NAME} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/inferenceopv.cpp)
target_include_directories(${PROJECT_NAME}
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
      PUBLIC  ${OpenCV_INCLUDE_DIRS}    
      ${OPENVINO_DIR}/include
      ${OPENVINO_DIR}/include/ie
      ${OPENVINO_DIR}/include/ngraph
)

if(MSVC)
    set(OPENVINO_LIB "C:/OpenVINO/runtime/lib/intel64/Release/openvino.lib")
else()
    set(OPENVINO_LIB "/opt/intel/openvino/runtime/lib/intel64/libopenvino.so")    
endif()
target_link_libraries(${PROJECT_NAME}
      ${OpenCV_LIBS}
      ${OPENVINO_LIB}
    )

add_executable(${PROJECT_NAME}_test main.cpp)
target_link_libraries(${PROJECT_NAME}_test
            ${OpenCV_LIBS}
            ${PROJECT_NAME}
    )