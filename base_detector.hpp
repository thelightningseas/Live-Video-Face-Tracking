#pragma once

#include "platform.hpp"
#include <inference_engine.hpp>
#include <samples/slog.hpp>
#include <samples/ocv_common.hpp>

struct BaseDetector {
	InferenceEngine::ExecutableNetwork net;
	InferenceEngine::InferencePlugin * plugin;
	InferenceEngine::InferRequest::Ptr request;
	std::string topoName;
	std::string pathToModel;
	std::string deviceForInference;
	const int maxBatch;
	bool isBatchDynamic;
	const bool isAsync;
	mutable bool enablingChecked;
	mutable bool _enabled;

	BaseDetector(std::string topoName,
		const std::string &pathToModel,
		const std::string &deviceForInference,
		int maxBatch, bool isBatchDynamic, bool isAsync);

	virtual ~BaseDetector();

	InferenceEngine::ExecutableNetwork* operator ->();
	virtual InferenceEngine::CNNNetwork read() = 0;
	virtual void submitRequest();
	virtual void wait();
	virtual InferenceEngine::StatusCode status();
	bool enabled() const;
	void printPerformanceCounts();
};

struct LoadDetector {
	BaseDetector& detector;

	explicit LoadDetector(BaseDetector& detector);

	void into(InferenceEngine::InferencePlugin & plg, bool enable_dynamic_batch = false) const;
};
