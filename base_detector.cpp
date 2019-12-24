#include "platform.hpp"
#include "base_detector.hpp"

using namespace InferenceEngine;

BaseDetector::BaseDetector(std::string topoName,
	const std::string &pathToModel,
	const std::string &deviceForInference,
	int maxBatch, bool isBatchDynamic, bool isAsync)
	: topoName(topoName), pathToModel(pathToModel), deviceForInference(deviceForInference),
	maxBatch(maxBatch), isBatchDynamic(isBatchDynamic), isAsync(isAsync),
	enablingChecked(false), _enabled(false) {
	if (isAsync) {
		slog::info << "Use async mode for " << topoName << slog::endl;
	}
}

BaseDetector::~BaseDetector() {}

ExecutableNetwork* BaseDetector::operator ->() {
	return &net;
}

void BaseDetector::submitRequest() {
	if (!enabled() || request == nullptr) return;
	if (isAsync) {
		request->StartAsync();
	}
	else {
		request->Infer();
	}
}

void BaseDetector::wait() {
	if (!enabled() || !request || !isAsync)
		return;
	request->Wait(IInferRequest::WaitMode::RESULT_READY);
}

StatusCode BaseDetector::status() {
	if (!enabled() || !request || !isAsync)
		return StatusCode::GENERAL_ERROR;
	return request->Wait(IInferRequest::WaitMode::STATUS_ONLY);
}

bool BaseDetector::enabled() const {
	if (!enablingChecked) {
		_enabled = !pathToModel.empty();
		if (!_enabled) {
			slog::info << topoName << " DISABLED" << slog::endl;
		}
		enablingChecked = true;
	}
	return _enabled;
}

void BaseDetector::printPerformanceCounts() {
	if (!enabled()) {
		return;
	}
	slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
	::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
}

LoadDetector::LoadDetector(BaseDetector& detector) : detector(detector) {
}

void LoadDetector::into(InferencePlugin & plg, bool enable_dynamic_batch) const {
	if (detector.enabled()) {
		std::map<std::string, std::string> config;
		if (enable_dynamic_batch) {
			config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
		}
		detector.net = plg.LoadNetwork(detector.read(), config);
		detector.plugin = &plg;
	}
}