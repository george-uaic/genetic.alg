#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

enum class ImprovementStrategy 
{
    FirstImprovement,
    BestImprovement,
    WorstImprovement,
    SimulatedAnnealing
};

struct Config
{
    int dimensions;
    int bits_per_dim;
    int total_bits;
    int iterations;
    int threads_per_block;
    int blocks;
    double interval_lower;
    double interval_upper;
    int precision;
    double cached_value_for_conversion;
    ImprovementStrategy improvement_strategy;
};


std::vector<double> launchHillClimb(const Config& config);