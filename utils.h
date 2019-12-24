#pragma once

#include "platform.hpp"

class CallStat {
public:
	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

	CallStat();

	double getSmoothedDuration();
	double getTotalDuration();
	void calculateDuration();
	void setStartTime();

private:
	size_t _number_of_calls;
	double _total_duration;
	double _last_call_duration;
	double _smoothed_duration;
	std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
public:
	void start(const std::string& name);
	void finish(const std::string& name);
	CallStat& operator[](const std::string& name);

private:
	std::map<std::string, CallStat> _timers;
};
