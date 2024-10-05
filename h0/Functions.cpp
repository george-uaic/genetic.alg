#include "Functions.h"
#include <iostream>

double Rastrigin(std::vector<double> v)
{
	auto sum = [](double acc, double x) { return acc + x * x - 10 * cos(2 * std::numbers::pi * x); };
	return std::accumulate(v.begin(), v.end(), 10.0 * v.size(), sum);
}

double Michalewicz(std::vector<double> v)
{
	int i{ 1 };
	int m{ 10 };
	auto sum = [&](double acc, double x) { return acc + sin(x) * pow(sin((i++ * x * x) / std::numbers::pi), 2 * m); };
	return -std::accumulate(v.begin(), v.end(), 0.0, sum);
}
