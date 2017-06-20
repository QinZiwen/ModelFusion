add_executable(rgbd_orbslam2 rgbd_orbslam2.cpp)
target_link_libraries(rgbd_orbslam2 ${OpenCV_LIBS} libORB_SLAM2.so ${Pangolin_LIBRARIES})

set_target_properties(rgbd_orbslam2
    PROPERTIES DEBUG_POSTFIX "d" 
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" 
    RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/bin"
)