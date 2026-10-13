cmake_minimum_required(VERSION 3.18)
project(afd_cam_async_ccu_experimental LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AFD_CANN_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Matching CANN 9.0 installation")
if(NOT AFD_CANN_ROOT)
  message(FATAL_ERROR "Set AFD_CANN_ROOT or ASCEND_HOME_PATH to the matching CANN installation")
endif()

find_path(AFD_CCU_INCLUDE ccu_kernel.h
  PATHS "${AFD_CANN_ROOT}/aarch64-linux/pkg_inc/hcomm/ccu"
        "${AFD_CANN_ROOT}/pkg_inc/hcomm/ccu" NO_DEFAULT_PATH REQUIRED)
find_path(AFD_HCCL_INCLUDE hccl_res.h
  PATHS "${AFD_CANN_ROOT}/aarch64-linux/include/hccl"
        "${AFD_CANN_ROOT}/include/hccl" NO_DEFAULT_PATH REQUIRED)
find_library(AFD_HCOMM_LIBRARY hcomm
  PATHS "${AFD_CANN_ROOT}/aarch64-linux/lib64"
        "${AFD_CANN_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(afd_cam_async_ccu_experimental SHARED ccu_segment_kernel.cpp ccu_submission.cpp)
target_compile_definitions(afd_cam_async_ccu_experimental PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(afd_cam_async_ccu_experimental PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(afd_cam_async_ccu_experimental SYSTEM PUBLIC
  "${AFD_CCU_INCLUDE}" "${AFD_HCCL_INCLUDE}"
  "${AFD_CANN_ROOT}/aarch64-linux/include" "${AFD_CANN_ROOT}/include")
target_link_libraries(afd_cam_async_ccu_experimental PRIVATE "${AFD_HCOMM_LIBRARY}")
target_compile_options(afd_cam_async_ccu_experimental PRIVATE -Wall -Wextra -Werror)
target_link_options(afd_cam_async_ccu_experimental PRIVATE -Wl,--no-undefined)
