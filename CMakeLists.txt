cmake_minimum_required(VERSION 3.21)
project(CECILIA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_DEBUG "-O3")
set(CMAKE_CXX_FLAGS_RELEASE  "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Libraries:
find_package(Threads REQUIRED)


# Helper
add_executable(helper apps/helper.cpp)
target_link_libraries(helper PRIVATE Threads::Threads cryptopp::cryptopp)

# CECILIA - ppRKN
add_executable(proxy_rkn apps/rkn/proxy.cpp core/Party.h core/rkn.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h ${ENC_FILES})
target_link_libraries(proxy_rkn PRIVATE Threads::Threads Eigen3::Eigen cryptopp::cryptopp)

# include dependencies using git submodule:
add_subdirectory(dependencies/eigen)
add_subdirectory(dependencies/cryptopp-cmake)
