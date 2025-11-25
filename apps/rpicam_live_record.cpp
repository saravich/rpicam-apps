/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2025, Raspberry Pi (Trading) Ltd.
 * A lightweight combined preview and recording app built on rpicam-hello.
 */

 #include <chrono>

 #include "core/rpicam_encoder.hpp"
 #include "output/output.hpp"
 
 using namespace std::placeholders;
 
 static int colourspace_flags_for_codec(std::string const &codec)
 {
         if (codec == "mjpeg" || codec == "yuv420")
                 return RPiCamEncoder::FLAG_VIDEO_JPEG_COLOURSPACE;
         return RPiCamEncoder::FLAG_VIDEO_NONE;
 }
 
 static std::string default_output_file(VideoOptions const &options)
 {
         std::string extension = options.Get().codec;
         if (extension == "libav")
                 extension = "mp4";
         else if (extension == "yuv420")
                 extension = "yuv420";
         else if (extension == "mjpeg")
                 extension = "mjpeg";
         else
                 extension = "h264";
 
         return "live-record." + extension;
 }
 
 static void event_loop(RPiCamEncoder &app)
 {
         VideoOptions const *options = app.GetOptions();
         std::unique_ptr<Output> output{Output::Create(options)};
         app.SetEncodeOutputReadyCallback(std::bind(&Output::OutputReady, output.get(), _1, _2, _3, _4));
         app.SetMetadataReadyCallback(std::bind(&Output::MetadataReady, output.get(), _1));
 
         app.OpenCamera();
         app.ConfigureVideo(colourspace_flags_for_codec(options->Get().codec));
         app.StartEncoder();
         app.StartCamera();
 
         auto start_time = std::chrono::high_resolution_clock::now();
 
         for (unsigned int count = 0;; ++count)
         {
                 RPiCamEncoder::Msg msg = app.Wait();
                 if (msg.type == RPiCamApp::MsgType::Timeout)
                 {
                         LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
                         app.StopCamera();
                         app.StartCamera();
                         continue;
                 }
                 if (msg.type == RPiCamEncoder::MsgType::Quit)
                         return;
                 if (msg.type != RPiCamEncoder::MsgType::RequestComplete)
                         throw std::runtime_error("unrecognised message!");
 
                 auto now = std::chrono::high_resolution_clock::now();
                 bool timeout = options->Get().timeout && (now - start_time) > options->Get().timeout.value;
                 bool frameout = options->Get().frames && count >= options->Get().frames;
 
                 CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);
                 if (!app.EncodeBuffer(completed_request, app.VideoStream()))
                 {
                         // Keep waiting until the encoder is ready.
                         start_time = now;
                         count = 0;
                 }
                 app.ShowPreview(completed_request, app.VideoStream());
 
                 if (timeout || frameout)
                 {
                         app.StopCamera();
                         app.StopEncoder();
                         return;
                 }
         }
 }
 
 int main(int argc, char *argv[])
 {
         try
         {
                 RPiCamEncoder app;
                 VideoOptions *options = app.GetOptions();
                 if (options->Parse(argc, argv))
                 {
                         if (options->Get().output.empty())
                         {
                                 options->Get().output = default_output_file(*options);
                                 LOG(1, "No output supplied, defaulting to " << options->Get().output);
                         }
                         if (options->Get().verbose >= 2)
                                 options->Get().Print();
 
                         event_loop(app);
                 }
         }
         catch (std::exception const &e)
         {
                 LOG_ERROR("ERROR: *** " << e.what() << " ***");
                 return -1;
         }
         return 0;
 