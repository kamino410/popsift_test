cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(main CXX)

find_package(PopSift REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.53.0 REQUIRED COMPONENTS program_options system filesystem)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC PopSift::popsift ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

