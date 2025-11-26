rpicam_still = executable('rpicam-still', files('rpicam_still.cpp'),
                          include_directories : include_directories('..'),
                          dependencies: [libcamera_dep, boost_dep],
                          link_with : rpicam_app,
                          install : true)

rpicam_vid = executable('rpicam-vid', files('rpicam_vid.cpp'),
                        include_directories : include_directories('..'),
                        dependencies: [libcamera_dep, boost_dep],
                        link_with : rpicam_app,
                        install : true)

rpicam_hello = executable('rpicam-hello', files('rpicam_hello.cpp'),
                          include_directories : include_directories('..'),
                          dependencies: libcamera_dep,
                          link_with : rpicam_app,
                          install : true)

rpicam_raw = executable('rpicam-raw', files('rpicam_raw.cpp'),
                        include_directories : include_directories('..'),
                        dependencies: [libcamera_dep, boost_dep],
                        link_with : rpicam_app,
                        install : true)

rpicam_jpeg = executable('rpicam-jpeg', files('rpicam_jpeg.cpp'),
                         include_directories : include_directories('..'),
                         dependencies: [libcamera_dep, boost_dep],
                         link_with : rpicam_app,
                         install : true)

if enable_tflite
    rpicam_detect = executable('rpicam-detect', files('rpicam_detect.cpp'),
                               include_directories : include_directories('..'),
                               dependencies: [libcamera_dep, boost_dep],
                               link_with : rpicam_app,
                               install : true)
endif

# OpenCV streaming app (requires OpenCV)
opencv_app_dep = dependency('opencv4', required : false)
if opencv_app_dep.found()
    # AprilTag OpenCV is optional - add it if available
    rpicam_opencv_stream_deps = [libcamera_dep, opencv_app_dep]
    
    # Check if AprilTag OpenCV integration is available
    if enable_apriltag_opencv and apriltag_opencv_dep.found()
        rpicam_opencv_stream_deps += apriltag_opencv_dep
    endif
    
    # C++ compiler arguments for rpicam-opencv-stream
    # Disable pedantic warnings when using AprilTag (flexible array members)
    rpicam_opencv_stream_cpp_args = cpp_arguments
    if enable_apriltag_opencv and apriltag_opencv_dep.found()
        rpicam_opencv_stream_cpp_args += ['-Wno-pedantic']
    endif
    
    # Build options for rpicam-opencv-stream
    rpicam_opencv_stream_options = []
    if enable_apriltag_opencv and apriltag_opencv_dep.found()
        rpicam_opencv_stream_options += ['werror=false']  # Disable -Werror when using AprilTag
    endif
    
    # Set rpath to find AprilTag libraries in build directory
    rpicam_opencv_stream_build_rpath = ''
    if enable_apriltag_opencv and apriltag_opencv_dep.found()
        # Add rpath to find AprilTag libraries in the build directory
        # meson.current_build_dir() gives the build directory for this subdir (apps/)
        # We need to go up one level to get to the root build dir, then into AprilTag
        apriltag_build_dir = meson.current_build_dir() / '..' / 'AprilTag'
        rpicam_opencv_stream_build_rpath = apriltag_build_dir
    endif
    
    rpicam_opencv_stream = executable('rpicam-opencv-stream', files('rpicam_opencv_stream.cpp'),
                                      include_directories : include_directories('..'),
                                      dependencies: rpicam_opencv_stream_deps,
                                      cpp_args : rpicam_opencv_stream_cpp_args,
                                      override_options : rpicam_opencv_stream_options,
                                      link_with : rpicam_app,
                                      build_rpath : rpicam_opencv_stream_build_rpath,
                                      install_rpath : get_option('prefix') / get_option('libdir'),
                                      install : true)
endif
