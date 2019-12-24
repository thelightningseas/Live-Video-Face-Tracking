#include "platform.hpp"
#include "face_detector.hpp"

using namespace InferenceEngine;

FaceDetector::FaceDetector(const std::string &pathToModel,
	const std::string &deviceForInference,
	int maxBatch, bool isBatchDynamic, bool isAsync,
	double detectionThreshold, bool doRawOutputMessages)
	: BaseDetector("Face Detection", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync),
	detectionThreshold(detectionThreshold), doRawOutputMessages(doRawOutputMessages),
	enquedFrames(0), width(0), height(0), bb_enlarge_coefficient(1.2), resultsFetched(false) {
}

void FaceDetector::submitRequest() {
	if (!enquedFrames) return;
	enquedFrames = 0;
	resultsFetched = false;
	results.clear();
	BaseDetector::submitRequest();
}

void FaceDetector::enqueue(const cv::Mat &frame) {
	if (!enabled()) return;

	if (!request) {
		request = net.CreateInferRequestPtr();
	}

	width = frame.cols;
	height = frame.rows;

	Blob::Ptr  inputBlob = request->GetBlob(input);

	matU8ToBlob<uint8_t>(frame, inputBlob);

	enquedFrames = 1;
}

CNNNetwork FaceDetector::read() {
	slog::info << "Loading network files for Face Detection" << slog::endl;
	CNNNetReader netReader;
	/** Read network model **/
	netReader.ReadNetwork(pathToModel);
	/** Set batch size to 1 **/
	slog::info << "Batch size is set to " << maxBatch << slog::endl;
	netReader.getNetwork().setBatchSize(maxBatch);
	/** Extract model name and load its weights **/
	std::string binFileName = fileNameNoExt(pathToModel) + ".bin";
	netReader.ReadWeights(binFileName);
	/** Read labels (if any)**/
	std::string labelFileName = fileNameNoExt(pathToModel) + ".labels";

	std::ifstream inputFile(labelFileName);
	std::copy(std::istream_iterator<std::string>(inputFile),
		std::istream_iterator<std::string>(),
		std::back_inserter(labels));
	// -----------------------------------------------------------------------------------------------------

	/** SSD-based network should have one input and one output **/
	// ---------------------------Check inputs -------------------------------------------------------------
	slog::info << "Checking Face Detection network inputs" << slog::endl;
	InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
	if (inputInfo.size() != 1) {
		throw std::logic_error("Face Detection network should have only one input");
	}
	InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
	inputInfoFirst->setPrecision(Precision::U8);
	// -----------------------------------------------------------------------------------------------------

	// ---------------------------Check outputs ------------------------------------------------------------
	slog::info << "Checking Face Detection network outputs" << slog::endl;
	OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
	if (outputInfo.size() != 1) {
		throw std::logic_error("Face Detection network should have only one output");
	}
	DataPtr& _output = outputInfo.begin()->second;
	output = outputInfo.begin()->first;

	const CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
	if (outputLayer->type != "DetectionOutput") {
		throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
			") should be DetectionOutput, but was " + outputLayer->type);
	}

	if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
		throw std::logic_error("Face Detection network output layer (" +
			output + ") should have num_classes integer attribute");
	}

	const int num_classes = outputLayer->GetParamAsInt("num_classes");
	if (labels.size() != num_classes) {
		if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, which has no label
			labels.insert(labels.begin(), "fake");
		else
			labels.clear();
	}
	const SizeVector outputDims = _output->getTensorDesc().getDims();
	maxProposalCount = outputDims[2];
	objectSize = outputDims[3];
	if (objectSize != 7) {
		throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
	}
	if (outputDims.size() != 4) {
		throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
			std::to_string(outputDims.size()));
	}
	_output->setPrecision(Precision::FP32);

	slog::info << "Loading Face Detection model to the " << deviceForInference << " plugin" << slog::endl;
	input = inputInfo.begin()->first;
	return netReader.getNetwork();
}

void FaceDetector::fetchResults() {
	if (!enabled()) return;
	results.clear();
	if (resultsFetched) return;
	resultsFetched = true;
	const float *detections = request->GetBlob(output)->buffer().as<float *>();

	for (int i = 0; i < maxProposalCount; i++) {
		float image_id = detections[i * objectSize + 0];
		Result r;
		r.label = static_cast<int>(detections[i * objectSize + 1]);
		r.confidence = detections[i * objectSize + 2];

		if (r.confidence <= detectionThreshold) {
			continue;
		}

		r.location.x = detections[i * objectSize + 3] * width;
		r.location.y = detections[i * objectSize + 4] * height;
		r.location.width = detections[i * objectSize + 5] * width - r.location.x;
		r.location.height = detections[i * objectSize + 6] * height - r.location.y;

		// Make square and enlarge face bounding box for more robust operation of face analytics networks
		int bb_width = r.location.width;
		int bb_height = r.location.height;

		int bb_center_x = r.location.x + bb_width / 2;
		int bb_center_y = r.location.y + bb_height / 2;

		int max_of_sizes = std::max(bb_width, bb_height);

		int bb_new_width = bb_enlarge_coefficient * max_of_sizes;
		int bb_new_height = bb_enlarge_coefficient * max_of_sizes;

		r.location.x = bb_center_x - bb_new_width / 2;
		r.location.y = bb_center_y - bb_new_height / 2;

		r.location.width = bb_new_width;
		r.location.height = bb_new_height;

		if (image_id < 0) {
			break;
		}
		if (doRawOutputMessages) {
			std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
				"    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
				<< r.location.height << ")"
				<< ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
		}

		results.push_back(r);
	}
}
