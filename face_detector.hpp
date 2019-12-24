#pragma once

#include "platform.hpp"
#include "base_detector.hpp"

struct FaceDetector : BaseDetector {
	struct Result {
		int label;
		float confidence;
		cv::Rect location;
	};

	std::string input;
	std::string output;
	double detectionThreshold;
	bool doRawOutputMessages;
	int maxProposalCount;
	int objectSize;
	int enquedFrames;
	float width;
	float height;
	const float bb_enlarge_coefficient;
	bool resultsFetched;
	std::vector<std::string> labels;
	std::vector<Result> results;

	FaceDetector(const std::string &pathToModel,
		const std::string &deviceForInference,
		int maxBatch, bool isBatchDynamic, bool isAsync,
		double detectionThreshold, bool doRawOutputMessages);

	InferenceEngine::CNNNetwork read() override;
	void submitRequest() override;

	void enqueue(const cv::Mat &frame);
	void fetchResults();
};
