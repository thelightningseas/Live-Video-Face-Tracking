#include "platform.hpp"
#include <gflags/gflags.h>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <ie_iextension.h>
#include <ext_list.hpp>

#include "utils.h"
#include "cam_stream.hpp"
#include "face_detector.hpp"

using namespace InferenceEngine;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}


int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        const bool isCamera = FLAGS_i == "cam";
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // ---------------------------------------------------------------------------------------------------
        // --------------------------- 1. Loading plugin to the Inference Engine -----------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}
        };
        FaceDetector faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r);
 
        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Loading extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Loading extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            pluginsForDevices[deviceName] = plugin;
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        // Disable dynamic batching for face detector as it processes one image at a time
        LoadDetector(faceDetector).into(pluginsForDevices[FLAGS_d], false);
        // ----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Doing inference -----------------------------------------------------
		// Starting inference & calculating performance
        slog::info << "Start inference " << slog::endl;
        if (!FLAGS_no_show) {
            std::cout << "Press any key to stop" << std::endl;
        }

        Timer timer;
        timer.start("total");

        std::ostringstream out;
        size_t framesCounter = 0; // possible overflow
        bool frameReadStatus;
        bool isLastFrame;
        cv::Mat prev_frame, next_frame, detect_frame, prev_detect_frame;

		// read input (video) frame
		cv::Mat frame;
		if (!cap.read(frame)) {
			throw std::logic_error("Failed to get frame from cv::VideoCapture");
		}

        // Detecting all faces on the first frame and reading the next one
        timer.start("detection");
		detect_frame = frame;
        faceDetector.enqueue(frame);
        faceDetector.submitRequest();
        timer.finish("detection");

        prev_frame = frame.clone();

        // Reading the next frame
        timer.start("video frame decoding");
        frameReadStatus = cap.read(frame);
        timer.finish("video frame decoding");

		std::vector<FaceDetector::Result> prev_detection_results, prev_prev_detection_results;

		std::deque<cv::Mat> frame_queue;
		std::vector<cv::Point2f> feature_points;

		timer.start("keypoints");
		timer.finish("keypoints");

		timer.start("tracker");
		timer.finish("tracker");

        while (true) {
			framesCounter++;
            isLastFrame = !frameReadStatus;

            // Retrieving face detection results for the previous frame
			if (faceDetector.status() == StatusCode::OK && framesCounter % 30 == 0) {
				timer.start("detection");
				faceDetector.wait();
				faceDetector.fetchResults();
				prev_detection_results = faceDetector.results;

				// No valid frame to infer if previous frame is the last
				if (!isLastFrame) {
					faceDetector.enqueue(frame);
					faceDetector.submitRequest();

					prev_detect_frame = detect_frame;
					detect_frame = frame;
				}
				timer.finish("detection");
			} else {
				frame_queue.push_back(frame);
			}

            // Reading the next frame if the current on e is not the last
            if (!isLastFrame) {
                timer.start("video frame decoding");
                frameReadStatus = cap.read(next_frame);
                timer.finish("video frame decoding");
            }

			// Track points

			// Update keypoints
			if (framesCounter % 30 == 0) {
				if (prev_detection_results.size() > 0) {
					timer.start("keypoints");
					cv::Mat mask(cv::Size(frame.cols, frame.rows), CV_8UC1, cv::Scalar(0));
					for (auto &result : prev_detection_results) {
						cv::Rect loc = result.location;
						cv::rectangle(mask, loc, cv::Scalar(255), -1);
					}

					cv::imshow("prev_detect_frame", prev_detect_frame);
					cv::imshow("detect_frame", detect_frame);

					cv::Mat frame_gray;
					cv::cvtColor(prev_detect_frame, frame_gray, cv::COLOR_BGR2GRAY);

					feature_points.clear();
					cv::goodFeaturesToTrack(frame_gray, feature_points, 50 * prev_detection_results.size(), 0.01, 10, mask, 3, 3);
					timer.finish("keypoints");

					// Track points
					timer.start("tracker");
					frame_queue.push_front(prev_detect_frame);
					frame_queue.push_back(detect_frame);
					for (int i = 0; i < frame_queue.size() - 1; i++) {
						if (feature_points.size() > 0) {
							std::vector<unsigned char> status;
							std::vector<float> err;
							cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

							std::vector<cv::Point2f> feature_points_next, feature_points_rev;
							cv::calcOpticalFlowPyrLK(frame_queue[i], frame_queue[i + 1], feature_points, feature_points_next,
								status, err, cv::Size(9, 9), 0, termcrit);
							cv::calcOpticalFlowPyrLK(frame_queue[i + 1], frame_queue[i], feature_points_next, feature_points_rev,
								status, err, cv::Size(9, 9), 0, termcrit);

							std::vector<cv::Point2f> good_points;
							for (int i = 0; i < feature_points.size(); i++) {
								float diff_x = abs(feature_points[i].x - feature_points_rev[i].x);
								float diff_y = abs(feature_points[i].y - feature_points_rev[i].y);
								if (MAX(diff_x, diff_y) <= 1.0) {
									good_points.push_back(feature_points_next[i]);
								}
							}

							cv::Point2f mean_pos;
							for (auto &point : good_points) {
								mean_pos += point / float(good_points.size());
							}
							
							for (auto &box : prev_detection_results) {
								box.location.x = mean_pos.x - box.location.width / 2;
								box.location.y = mean_pos.y - box.location.height / 2;
							}

							//cv::Mat vis_frame = frame.clone();
							//cv::circle(vis_frame, mean_pos, 4, cv::Scalar(0, 0, 255), -1);
							//cv::imshow("vis_frame", vis_frame);

							feature_points = good_points;
						}

						// Track box
						//cv::Mat vis_frame = frame_queue[i + 1].clone();
						//for (auto &point : feature_points) {
						//	cv::circle(vis_frame, point, 2, cv::Scalar(255, 255, 0), -1);
						//}
						//cv::imshow("vis", vis_frame);
						//cv::waitKey(0);
					}
					frame_queue.clear();
					timer.finish("tracker");
				}
			} else {
				// Track points
				if (feature_points.size() > 0) {
					std::vector<unsigned char> status;
					std::vector<float> err;
					cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

					std::vector<cv::Point2f> feature_points_next, feature_points_rev;
					cv::calcOpticalFlowPyrLK(prev_frame, frame, feature_points, feature_points_next,
						status, err, cv::Size(9, 9), 3, termcrit);
					cv::calcOpticalFlowPyrLK(frame, prev_frame, feature_points_next, feature_points_rev,
						status, err, cv::Size(9, 9), 3, termcrit);

					std::vector<cv::Point2f> good_points;
					for (int i = 0; i < feature_points.size(); i++) {
						float diff_x = abs(feature_points[i].x - feature_points_rev[i].x);
						float diff_y = abs(feature_points[i].y - feature_points_rev[i].y);
						if (MAX(diff_x, diff_y) <= 1.0) {
							good_points.push_back(feature_points_next[i]);
						}
					}

					cv::Point2f mean_pos;
					for (auto &point : feature_points) {
						mean_pos += point / float(feature_points.size());
					}

					for (auto &box : prev_detection_results) {
						box.location.x = mean_pos.x - box.location.width / 2;
						box.location.y = mean_pos.y - box.location.height / 2;
					}

					feature_points = good_points;
				}
			}

			//if (prev_prev_detection_results.size() > 0) {
			//	float diff = prev_detection_results[0].location.x - prev_prev_detection_results[0].location.x;
			//	std::cout << "x diff: " << abs(diff) / prev_detection_results[0].location.width << std::endl;
			//}
			//prev_prev_detection_results = prev_detection_results;

            // Visualizing results
            if (!FLAGS_no_show) {
                timer.start("visualization");
				cv::Mat vis_frame = frame.clone();

                out.str("");
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (timer["video frame decoding"].getSmoothedDuration() +
                       timer["visualization"].getSmoothedDuration())
                    << " ms";
                cv::putText(vis_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                            cv::Scalar(0, 255, 0));

                out.str("");
                out << "Keypoint detection time: " << std::fixed << std::setprecision(2)
                    << timer["tracker"].getSmoothedDuration()
                    << " ms ("
                    << 1000.f / (timer["tracker"].getSmoothedDuration())
                    << " fps)";
                cv::putText(vis_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                            cv::Scalar(0, 255, 0));

                // For every detected face
                int i = 0;
                for (auto &result : prev_detection_results) {
                    cv::Rect rect = result.location;

                    out.str("");

                    out << (result.label < faceDetector.labels.size() ? faceDetector.labels[result.label] :
                            std::string("label #") + std::to_string(result.label))
                        << ": " << std::fixed << std::setprecision(3) << result.confidence;

                    cv::putText(vis_frame,
                                out.str(),
                                cv::Point2f(result.location.x, result.location.y - 15),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                0.8,
                                cv::Scalar(0, 0, 255));

                    cv::rectangle(vis_frame, result.location, cv::Scalar(100, 100, 100), 1);
                    i++;
                }

				// For every feature point
				for (auto &point : feature_points) {
					cv::circle(vis_frame, point, 2, cv::Scalar(255, 255, 0), -1);
				}

                cv::imshow("Detection results", vis_frame);
                timer.finish("visualization");
            }

            // End of file (or a single frame file like an image). The last frame is displayed to let you check what is shown
            if (isLastFrame) {
                timer.finish("total");
                if (!FLAGS_no_wait) {
                    std::cout << "No more frames to process. Press any key to exit" << std::endl;
                    cv::waitKey(0);
                }
                break;
            } else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
                timer.finish("total");
                break;
            }

            prev_frame = frame;
            frame = next_frame;
            next_frame = cv::Mat();
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Showing performance results
        if (FLAGS_pc) {
            faceDetector.printPerformanceCounts();
        }
        // ---------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
