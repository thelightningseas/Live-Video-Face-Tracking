#include "platform.hpp"
#include "utils.h"

CallStat::CallStat() :
	_number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) {
}

double CallStat::getSmoothedDuration() {
	// Additional check is needed for the first frame while duration of the first
	// visualisation is not calculated yet.
	if (_smoothed_duration < 0) {
		auto t = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<ms>(t - _last_call_start).count();
	}
	return _smoothed_duration;
}

double CallStat::getTotalDuration() {
	return _total_duration;
}

void CallStat::calculateDuration() {
	auto t = std::chrono::high_resolution_clock::now();
	_last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
	_number_of_calls++;
	_total_duration += _last_call_duration;
	if (_smoothed_duration < 0) {
		_smoothed_duration = _last_call_duration;
	}
	double alpha = 0.1;
	_smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
}

void CallStat::setStartTime() {
	_last_call_start = std::chrono::high_resolution_clock::now();
}


void Timer::start(const std::string& name) {
	if (_timers.find(name) == _timers.end()) {
		_timers[name] = CallStat();
	}
	_timers[name].setStartTime();
}

void Timer::finish(const std::string& name) {
	auto& timer = (*this)[name];
	timer.calculateDuration();
}

CallStat& Timer::operator[](const std::string& name) {
	if (_timers.find(name) == _timers.end()) {
		throw std::logic_error("No timer with name " + name + ".");
	}
	return _timers[name];
}
