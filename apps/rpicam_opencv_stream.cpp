/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2024, Raspberry Pi (Trading) Ltd.
 *
 * rpicam_opencv_stream.cpp - Stream camera feed using OpenCV
 */

#include <chrono>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>

#include "apriltag_opencv.h"
#include "apriltag_family.h"

#include "core/rpicam_app.hpp"
#include "core/options.hpp"
#include "core/buffer_sync.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

// Pose estimation parameters (can be set via command line)
struct PoseEstimationParams {
	double tag_size = 0.1;  // Tag size in meters (default: 10cm)
	double fx = 0.0;        // Focal length X (0 = auto-estimate)
	double fy = 0.0;        // Focal length Y (0 = auto-estimate)
	double cx = 0.0;        // Principal point X (0 = auto-estimate)
	double cy = 0.0;        // Principal point Y (0 = auto-estimate)
	bool enabled = true;   // Enable pose estimation
};


// Initialize pose estimation parameters
// Edit these values directly in the code to set camera parameters
PoseEstimationParams init_pose_params()
{
	PoseEstimationParams params;
	
	// ============================================
	// CAMERA PARAMETERS - EDIT THESE VALUES
	// ============================================
	
	// Tag size in meters (physical size of your AprilTag)
	params.tag_size = 0.14;  // e.g., 0.14 = 14cm
	
	// Camera intrinsic parameters
	// Set to 0.0 to auto-estimate from image dimensions
	params.fx = 2719.0;  // Focal length X in pixels
	params.fy = 2716.0;  // Focal length Y in pixels
	params.cx = 1640.0;  // Principal point X (0 = image center)
	params.cy = 1232.0;  // Principal point Y (0 = image center)
	
	// Enable/disable pose estimation
	params.enabled = true;
	
	// ============================================
	
	return params;
}

// The main event loop for the application.
static void event_loop(RPiCamApp &app, const PoseEstimationParams &pose_params)
{
	Options const *options = app.GetOptions();

	app.OpenCamera();
	app.ConfigureViewfinder();
	app.StartCamera();

	auto start_time = std::chrono::high_resolution_clock::now();

	// Get stream information
	StreamInfo info;
	libcamera::Stream *stream = app.ViewfinderStream(&info);
	if (!stream)
		throw std::runtime_error("Viewfinder stream not available");

	LOG(1, "Stream info: " << info.width << "x" << info.height << ", stride=" << info.stride);

	// Validate dimensions are even (required for YUV420)
	if (info.width % 2 != 0 || info.height % 2 != 0)
	{
		throw std::runtime_error("Stream dimensions must be even for YUV420 format");
	}

	// Get display window size from options
	// Use viewfinder dimensions if set, otherwise use camera stream dimensions
	unsigned int display_width = info.width;
	unsigned int display_height = info.height;
	
	// Check if user wants a different display size than the stream
	// We'll use preview_width/height if available, otherwise viewfinder dimensions
	if (options->Get().preview_width > 0 && options->Get().preview_height > 0)
	{
		display_width = options->Get().preview_width;
		display_height = options->Get().preview_height;
		LOG(1, "Using preview window size for display: " << display_width << "x" << display_height);
	}
	else if (options->Get().viewfinder_width > 0 && options->Get().viewfinder_height > 0)
	{
		// viewfinder dimensions already match stream, but we can still resize display
		display_width = options->Get().viewfinder_width;
		display_height = options->Get().viewfinder_height;
		LOG(1, "Using viewfinder size for display: " << display_width << "x" << display_height);
	}
	else
	{
		LOG(1, "Display window size matches stream: " << display_width << "x" << display_height);
		LOG(1, "Use --preview <width>,<height> or --viewfinder-width/height to resize");
	}

	// Create OpenCV window (check for display first)
	const std::string window_name = "Camera Stream";
	try
	{
		// Create window with specific size (or WINDOW_AUTOSIZE if using stream dimensions)
		if (display_width == info.width && display_height == info.height)
		{
			namedWindow(window_name, WINDOW_AUTOSIZE);
		}
		else
		{
			namedWindow(window_name, WINDOW_NORMAL);
			resizeWindow(window_name, display_width, display_height);
		}
		
		// Try to show a test image to verify display works
		Mat test_img(100, 100, CV_8UC3, Scalar(0, 0, 0));
		imshow(window_name, test_img);
		waitKey(1); // Process window events
		LOG(1, "OpenCV window created successfully");
	}
	catch (const cv::Exception &e)
	{
		LOG_ERROR("Failed to create OpenCV window: " << e.what());
		LOG_ERROR("Make sure you have a display available (DISPLAY environment variable set)");
		throw;
	}

	LOG(1, "Streaming camera feed. Press 'q' or ESC to quit.");
	LOG(1, "Window will stay open until you press 'q' or ESC, or timeout is reached.");

	// Initialize AprilTag detector
	apriltag_family_t *tf = apriltag_family_create("tag36h11");
	if (!tf)
	{
		LOG_ERROR("Failed to create AprilTag family (tag36h11)");
		throw std::runtime_error("AprilTag initialization failed");
	}

	apriltag_detector_t *td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);
	
	// Configure detector (adjust these as needed)
	td->quad_decimate = 2.0;  // Decimate input image for faster processing
	td->quad_sigma = 0.0;     // Apply low-pass blur (0 = disabled)
	td->nthreads = 4;         // Number of threads
	td->debug = 0;            // Debug mode
	td->refine_edges = 1;     // Spend more time aligning edges
	td->refine_decode = 0;    // Spend more time decoding tags
	td->refine_pose = 0;      // Spend more time computing pose

	LOG(1, "AprilTag detector initialized (tag36h11 family)");

	// Pose estimation setup
	Mat camera_matrix;
	Mat dist_coeffs = Mat::zeros(4, 1, CV_64F);
	std::vector<Point3d> object_points;
	
	if (pose_params.enabled)
	{
		// Camera intrinsic parameters
		// Use provided values or auto-estimate from image dimensions
		double fx = pose_params.fx > 0 ? pose_params.fx : static_cast<double>(display_width);
		double fy = pose_params.fy > 0 ? pose_params.fy : static_cast<double>(display_height);
		double cx = pose_params.cx > 0 ? pose_params.cx : static_cast<double>(display_width) / 2.0;
		double cy = pose_params.cy > 0 ? pose_params.cy : static_cast<double>(display_height) / 2.0;
		
		// Camera matrix
		camera_matrix = (Mat_<double>(3, 3) <<
			fx, 0, cx,
			0, fy, cy,
			0, 0, 1);
		
		// 3D object points for AprilTag (in tag coordinate system)
		// Tag corners in 3D space: center at origin, tag in XY plane
		// Standard AprilTag: corners at (-1,-1), (1,-1), (1,1), (-1,1) in tag units
		// We scale by tag_size/2 to get actual dimensions
		double half_size = pose_params.tag_size / 2.0;
		object_points.push_back(Point3d(-half_size, -half_size, 0));  // Bottom-left
		object_points.push_back(Point3d(half_size, -half_size, 0));   // Bottom-right
		object_points.push_back(Point3d(half_size, half_size, 0));     // Top-right
		object_points.push_back(Point3d(-half_size, half_size, 0));    // Top-left
		
		LOG(1, "Pose estimation enabled");
		LOG(1, "Tag size: " << pose_params.tag_size << "m");
		LOG(1, "Camera intrinsics: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy);
	}
	else
	{
		LOG(1, "Pose estimation disabled");
	}

	// Skip first frame to ensure camera is fully initialized
	bool first_frame = true;
	bool should_quit = false;

	for (unsigned int count = 0; !should_quit; count++)
	{
		try
		{
			RPiCamApp::Msg msg = app.Wait();
			if (msg.type == RPiCamApp::MsgType::Timeout)
			{
				LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
				app.StopCamera();
				app.StartCamera();
				continue;
			}
			if (msg.type == RPiCamApp::MsgType::Quit)
			{
				LOG(1, "Quit message received");
				break;
			}
			else if (msg.type != RPiCamApp::MsgType::RequestComplete)
			{
				LOG_ERROR("Unrecognised message type: " << static_cast<int>(msg.type));
				continue;
			}

			auto now = std::chrono::high_resolution_clock::now();
			if (options->Get().timeout && (now - start_time) > options->Get().timeout.value)
			{
				LOG(1, "Timeout reached");
				break;
			}

		CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);

		// Check if buffer exists for this stream
		if (completed_request->buffers.find(stream) == completed_request->buffers.end())
		{
			LOG_ERROR("No buffer found for stream");
			continue;
		}

		// Skip first frame to ensure everything is initialized
		if (first_frame)
		{
			LOG(1, "Skipping first frame for initialization");
			first_frame = false;
			continue;
		}

		// Read frame buffer - keep BufferReadSync alive for the entire frame processing
		BufferReadSync r(&app, completed_request->buffers[stream]);
		const std::vector<libcamera::Span<uint8_t>> mem = r.Get();
		
		if (mem.empty() || mem[0].empty())
		{
			LOG_ERROR("Empty frame buffer");
			continue;
		}

		// Validate buffer size - YUV420 needs Y plane + U plane + V plane
		// Y plane: stride * height
		// U plane: (stride/2) * (height/2)
		// V plane: (stride/2) * (height/2)
		size_t y_size = info.stride * info.height;
		size_t uv_size = (info.stride / 2) * (info.height / 2);
		size_t expected_size = y_size + uv_size * 2;
		
		if (mem[0].size() < expected_size)
		{
			LOG_ERROR("Frame buffer too small: got " << mem[0].size() << ", expected " << expected_size 
			          << " (Y:" << y_size << " U:" << uv_size << " V:" << uv_size << ")");
			continue;
		}

		// Validate dimensions
		if (info.width == 0 || info.height == 0 || info.stride == 0)
		{
			LOG_ERROR("Invalid stream dimensions: " << info.width << "x" << info.height << " stride=" << info.stride);
			continue;
		}

		// Convert YUV420 to BGR for OpenCV display
		// YUV420 format: Y plane (full size) + U plane (quarter size) + V plane (quarter size)
		const uint8_t *y_plane = mem[0].data();
		const uint8_t *u_plane = y_plane + y_size;
		const uint8_t *v_plane = u_plane + uv_size;
		
		// Validate pointers are within buffer bounds
		if (u_plane >= mem[0].data() + mem[0].size() || v_plane >= mem[0].data() + mem[0].size())
		{
			LOG_ERROR("Invalid plane pointers - buffer overflow");
			continue;
		}

		// Create BGR output Mat
		Mat bgr_frame;

		try
		{
			// Create BGR output Mat directly
			bgr_frame = Mat(info.height, info.width, CV_8UC3);
			
			// Manual YUV420 to BGR conversion
			// YUV420: Y plane full size, U and V planes quarter size
			for (unsigned int y = 0; y < info.height; y++)
			{
				uint8_t *bgr_row = bgr_frame.ptr(y);
				const uint8_t *y_row = y_plane + y * info.stride;
				
				for (unsigned int x = 0; x < info.width; x++)
				{
					// Get Y, U, V values
					int Y = y_row[x];
					
					// U and V are subsampled (quarter resolution)
					unsigned int uv_x = x / 2;
					unsigned int uv_y = y / 2;
					const uint8_t *u_row = u_plane + uv_y * (info.stride / 2);
					const uint8_t *v_row = v_plane + uv_y * (info.stride / 2);
					
					int U = u_row[uv_x] - 128; // Center U around 0
					int V = v_row[uv_x] - 128; // Center V around 0
					
					// Convert YUV to BGR using standard formulas
					int R = Y + (int)(1.402 * V);
					int G = Y - (int)(0.344 * U) - (int)(0.714 * V);
					int B = Y + (int)(1.772 * U);
					
					// Clamp to valid range
					R = std::max(0, std::min(255, R));
					G = std::max(0, std::min(255, G));
					B = std::max(0, std::min(255, B));
					
					// BGR format (OpenCV uses BGR, not RGB)
					bgr_row[x * 3 + 0] = B;
					bgr_row[x * 3 + 1] = G;
					bgr_row[x * 3 + 2] = R;
				}
			}
			
			// Validate conversion succeeded
			if (bgr_frame.empty() || bgr_frame.rows != static_cast<int>(info.height) || bgr_frame.cols != static_cast<int>(info.width))
			{
				LOG_ERROR("BGR conversion failed or wrong size");
				continue;
			}
		}
		catch (const cv::Exception &e)
		{
			LOG_ERROR("OpenCV error: " << e.what());
			continue;
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error converting frame: " << e.what());
			continue;
		}

		// Display the frame
		if (bgr_frame.empty())
		{
			LOG_ERROR("Empty BGR frame after conversion");
			continue;
		}

		try
		{
			// Resize frame if display size differs from stream size
			Mat display_frame;
			if (display_width != info.width || display_height != info.height)
			{
				resize(bgr_frame, display_frame, Size(display_width, display_height), 0, 0, INTER_LINEAR);
			}
			else
			{
				display_frame = bgr_frame;
			}

			// Convert to grayscale for AprilTag detection
			Mat gray;
			cvtColor(display_frame, gray, COLOR_BGR2GRAY);

			// Run AprilTag detection
			image_u8_t *im8 = cv2im8_copy(gray);
			zarray_t *detections = apriltag_detector_detect(td, im8);

			// Draw detections on the frame
			if (zarray_size(detections) > 0)
			{
				LOG(1, "Detected " << zarray_size(detections) << " AprilTag(s)");

				for (int i = 0; i < zarray_size(detections); i++)
				{
					apriltag_detection_t *det;
					zarray_get(detections, i, &det);

					// Draw tag outline (4 corners)
					for (int j = 0; j < 4; j++)
					{
						int next_j = (j + 1) % 4;
						cv::Point pt1(det->p[j][0], det->p[j][1]);
						cv::Point pt2(det->p[next_j][0], det->p[next_j][1]);
						line(display_frame, pt1, pt2, Scalar(0, 255, 0), 2);
					}

					// Draw tag center
					cv::Point center(det->c[0], det->c[1]);
					circle(display_frame, center, 5, Scalar(0, 0, 255), -1);

					// Draw tag ID
					std::string id_text = "ID: " + std::to_string(det->id);
					cv::Point text_pos(det->c[0] - 20, det->c[1] - 10);
					putText(display_frame, id_text, text_pos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
					putText(display_frame, id_text, text_pos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

					// Pose estimation
					if (pose_params.enabled && !object_points.empty())
					{
						// Convert detected corners to image points
						std::vector<Point2d> image_points;
						for (int j = 0; j < 4; j++)
						{
							image_points.push_back(Point2d(det->p[j][0], det->p[j][1]));
						}
						
						// Estimate pose using solvePnP
						Mat rvec, tvec;  // Rotation and translation vectors
						bool success = solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
						
						if (success)
						{
							// Draw 3D coordinate axes to visualize pose
							// Axes length: 30% of tag size
							double axis_length = pose_params.tag_size * 0.3;
							
							// 3D points for axes endpoints
							std::vector<Point3d> axis_points;
							axis_points.push_back(Point3d(0, 0, 0));           // Origin
							axis_points.push_back(Point3d(axis_length, 0, 0)); // X axis (red)
							axis_points.push_back(Point3d(0, axis_length, 0)); // Y axis (green)
							axis_points.push_back(Point3d(0, 0, -axis_length)); // Z axis (blue) - negative because camera looks down +Z
							
							// Project 3D points to 2D
							std::vector<Point2d> projected_points;
							projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);
							
							// Draw axes
							Point2d origin = projected_points[0];
							line(display_frame, origin, projected_points[1], Scalar(0, 0, 255), 3); // X axis - Red
							line(display_frame, origin, projected_points[2], Scalar(0, 255, 0), 3); // Y axis - Green
							line(display_frame, origin, projected_points[3], Scalar(255, 0, 0), 3); // Z axis - Blue
							
							// Draw axis labels
							putText(display_frame, "X", projected_points[1], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
							putText(display_frame, "Y", projected_points[2], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
							putText(display_frame, "Z", projected_points[3], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
							
							// Display pose information
							double distance = sqrt(tvec.at<double>(0)*tvec.at<double>(0) + 
							                       tvec.at<double>(1)*tvec.at<double>(1) + 
							                       tvec.at<double>(2)*tvec.at<double>(2));
							
							std::ostringstream dist_str;
							dist_str << std::fixed << std::setprecision(2) << distance;
							std::string pose_text = "D: " + dist_str.str() + "m";
							cv::Point pose_text_pos(det->c[0] - 30, det->c[1] + 20);
							putText(display_frame, pose_text, pose_text_pos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 2);
							
							LOG(2, "Tag ID: " << det->id << " at (" << det->c[0] << ", " << det->c[1] 
							    << "), distance: " << distance << "m");
						}
						else
						{
							LOG(2, "Tag ID: " << det->id << " at (" << det->c[0] << ", " << det->c[1] << ") - pose estimation failed");
						}
					}
				}
			}

			// Clean up detections
			apriltag_detections_destroy(detections);
			image_u8_destroy(im8);
			
			imshow(window_name, display_frame);

			// Check for 'q' key press to quit
			// waitKey returns -1 if no key is pressed, so we need to check for that
			int key_code = waitKey(1);
			if (key_code >= 0)
			{
				char key = key_code & 0xFF;
				if (key == 'q' || key == 'Q' || key == 27) // 27 is ESC key
				{
					LOG(1, "Quit requested by user (key: " << (int)key << ")");
					should_quit = true;
					break;
				}
			}
		}
		catch (const cv::Exception &e)
		{
			LOG_ERROR("OpenCV display error: " << e.what());
			// Don't exit on display errors, just continue to next frame
			continue;
		}

		LOG(2, "Frame " << count << " displayed");
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error in main loop: " << e.what());
			// Try to continue, but if it keeps failing, we'll exit
			continue;
		}
	}

	// Clean up AprilTag detector
	apriltag_detector_destroy(td);
	apriltag_family_destroy(tf);
	LOG(1, "AprilTag detector cleaned up");

	destroyWindow(window_name);
}

int main(int argc, char *argv[])
{
	try
	{
		// Initialize pose estimation parameters
		// Edit init_pose_params() function to set camera parameters
		PoseEstimationParams pose_params = init_pose_params();
		
		RPiCamApp app;
		Options *options = app.GetOptions();
		
		// Parse standard options
		if (options->Parse(argc, argv))
		{
			// Disable the default preview window since we're using OpenCV
			// This must be set before OpenCamera() is called
			options->Set().nopreview = true;
			
			if (options->Get().verbose >= 2)
				options->Get().Print();
			
			// Log pose estimation parameters
			if (pose_params.enabled)
			{
				LOG(1, "Pose estimation enabled");
				LOG(1, "Tag size: " << pose_params.tag_size << "m");
				LOG(1, "Camera fx: " << pose_params.fx << ", fy: " << pose_params.fy);
				LOG(1, "Camera cx: " << pose_params.cx << ", cy: " << pose_params.cy);
			}
			else
			{
				LOG(1, "Pose estimation disabled");
			}

			event_loop(app, pose_params);
		}
	}
	catch (std::exception const &e)
	{
		LOG_ERROR("ERROR: *** " << e.what() << " ***");
		return -1;
	}
	return 0;
}

