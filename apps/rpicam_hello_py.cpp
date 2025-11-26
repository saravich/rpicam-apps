/* SPDX-License-Identifier: BSD-2-Clause */

#include <chrono>
#include <stdexcept>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/rpicam_app.hpp"
#include "core/options.hpp"

namespace py = pybind11;
using namespace std::placeholders;

// Original event loop extracted from rpicam_hello.cpp
static void event_loop(RPiCamApp &app)
{
    Options const *options = app.GetOptions();

    app.OpenCamera();
    app.ConfigureViewfinder();
    app.StartCamera();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int count = 0;; count++)
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
            return;
        else if (msg.type != RPiCamApp::MsgType::RequestComplete)
            throw std::runtime_error("unrecognised message!");

        LOG(2, "Viewfinder frame " << count);
        auto now = std::chrono::high_resolution_clock::now();
        if (options->Get().timeout && (now - start_time) > options->Get().timeout.value)
            return;

        CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);
        app.ShowPreview(completed_request, app.ViewfinderStream());
    }
}

// C++ function we expose to Python.
// args is like ["--timeout", "5000", "--verbose", "2"]
int run_rpicam_hello(const std::vector<std::string> &args)
{
    try
    {
        // Build argv-style vector
        std::vector<std::string> all_args;
        all_args.reserve(args.size() + 1);
        all_args.push_back("rpicam_hello"); // program name
        all_args.insert(all_args.end(), args.begin(), args.end());

        std::vector<char *> argv;
        argv.reserve(all_args.size());
        for (auto &s : all_args)
            argv.push_back(const_cast<char *>(s.c_str()));

        int argc = static_cast<int>(argv.size());

        RPiCamApp app;
        Options *options = app.GetOptions();
        if (options->Parse(argc, argv.data()))
        {
            if (options->Get().verbose >= 2)
                options->Get().Print();

            // Release GIL while we block in the event loop
            py::gil_scoped_release release;
            event_loop(app);
        }
    }
    catch (std::exception const &e)
    {
        LOG_ERROR("ERROR: *** " << e.what() << " ***");
        return -1;
    }

    return 0;
}

// Helper: run loop and call Python callback for each frame
int run_rpicam_hello_stream(const std::vector<std::string> &args, py::function frame_cb)
{
    try
    {
        // Build argv-style vector (same as before)
        std::vector<std::string> all_args;
        all_args.reserve(args.size() + 1);
        all_args.push_back("rpicam_hello");
        all_args.insert(all_args.end(), args.begin(), args.end());

        std::vector<char *> argv;
        argv.reserve(all_args.size());
        for (auto &s : all_args)
            argv.push_back(const_cast<char *>(s.c_str()));
        int argc = static_cast<int>(argv.size());

        RPiCamApp app;
        Options *options = app.GetOptions();
        if (!options->Parse(argc, argv.data()))
            return 0;

        if (options->Get().verbose >= 2)
            options->Get().Print();

        app.OpenCamera();
        app.ConfigureViewfinder();
        app.StartCamera();

        auto start_time = std::chrono::high_resolution_clock::now();

        for (unsigned int count = 0;; count++)
        {
            // Release GIL while we block waiting for camera
            py::gil_scoped_release release;

            RPiCamApp::Msg msg = app.Wait();

            if (msg.type == RPiCamApp::MsgType::Timeout)
            {
                LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
                app.StopCamera();
                app.StartCamera();
                continue;
            }
            if (msg.type == RPiCamApp::MsgType::Quit)
                break;
            else if (msg.type != RPiCamApp::MsgType::RequestComplete)
                throw std::runtime_error("unrecognised message!");

            auto now = std::chrono::high_resolution_clock::now();
            if (options->Get().timeout && (now - start_time) > options->Get().timeout.value)
                break;

            CompletedRequestPtr &completed_request =
                std::get<CompletedRequestPtr>(msg.payload);

            // --------- map frame buffer to numpy array ----------

            libcamera::Stream *stream = app.ViewfinderStream();
            libcamera::FrameBuffer *buffer =
                completed_request->buffers.at(stream).get();

            // You’ll find mapping code in the existing repo, but roughly:
            // map first plane to memory
            const libcamera::FrameMetadata &meta = buffer->metadata();
            const libcamera::FrameMetadata::Plane &p = meta.planes()[0];

            uint8_t *data = buffer->planes()[0].data;
            int width  = stream->configuration().size.width;
            int height = stream->configuration().size.height;
            int stride = stream->configuration().stride;

            // Example assuming packed RGB888 with stride == width*3
            // Adjust shape/strides to *your actual format*.
            py::gil_scoped_acquire acquire;

            py::capsule owner(buffer, [](void *) {
                /* no-op or custom unmap if needed */
            });

            // shape: (height, width, 3)
            std::vector<ssize_t> shape  = { height, width, 3 };
            std::vector<ssize_t> strides = { stride, 3, 1 };

            py::array frame(
                py::buffer_info(
                    data,
                    sizeof(uint8_t),
                    py::format_descriptor<uint8_t>::format(),
                    3,
                    shape,
                    strides
                )
            );

            try {
                frame_cb(frame);   // call Python callback
            } catch (const py::error_already_set &e) {
                // allow Python to stop the loop by raising an exception
                LOG_ERROR("Python callback raised, stopping stream: " << e.what());
                break;
            }
        }

        app.StopCamera();
    }
    catch (std::exception const &e)
    {
        LOG_ERROR("ERROR: *** " << e.what() << " ***");
        return -1;
    }

    return 0;
}




// pybind11 module
PYBIND11_MODULE(rpicam_hello, m)
{
    m.doc() = "pybind11 wrapper for rpicam-hello";

    m.def(
        "run",
        &run_rpicam_hello,
        py::arg("args") = std::vector<std::string>{},
        "Run the rpicam-hello app with CLI-style args."
    );

    m.def(
        "run_stream",
        &run_rpicam_hello_stream,
        py::arg("args"),
        py::arg("frame_callback"),
        "Run camera loop and call frame_callback(frame) for each frame.\n"
        "frame is a NumPy array (H x W x C)."
    );
}
