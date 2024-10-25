#include "CudaHillClimbing.cuh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>

int main() 
{
    Config config{};
    config.interval_lower = -5.12;
    config.interval_upper = 5.12;
    config.precision = 5;
    config.dimensions = 30;
    double segments = (config.interval_upper - config.interval_lower) * pow(10, config.precision);
    config.bits_per_dim = (int)ceil(log2(segments));
    config.total_bits = config.dimensions * config.bits_per_dim;
    config.cached_value_for_conversion = (config.interval_upper - config.interval_lower) / ((1ull << config.bits_per_dim) - 1);

    config.iterations = 10000;
    config.threads_per_block = 32;
    config.blocks = (config.iterations + config.threads_per_block - 1) / config.threads_per_block;

    config.improvement_strategy = ImprovementStrategy::SimulatedAnnealing;

    std::cout << std::fixed << std::setprecision(config.precision);
    //size_t total_memory = 0;
    //total_memory += sizeof(bool) * config.iterations * config.total_bits;  // bitstrings
    //total_memory += sizeof(double) * config.iterations * config.dimensions;  // real_values
    //total_memory += sizeof(double) * config.iterations;  // fitness_values
    //total_memory += sizeof(curandState) * config.threads_per_block * config.blocks;  // random states
    //
    //std::cout << "Total memory to be allocated: " << total_memory / (1024.0 * 1024.0) << " MB" << '\n';
    //std::cout << "Grid size: " << config.blocks << ", Block size: " << config.threads_per_block << '\n';

    std::vector<double> results;
    std::vector<float> time;
    int samples = 30;
    for (int i = 1; i <= samples; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        std::vector<double> sample_results = launchHillClimb(config);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        results.push_back(*std::min_element(sample_results.begin(), sample_results.end()));
        time.push_back(milliseconds);
        std::cout << "sample " << i << ": " << results.back() << '\n';
    }

    auto min = std::min_element(results.begin(), results.end());
    double average = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
    float averageTime = std::accumulate(time.begin(), time.end(), 0.f) / time.size();
    
    std::cout << "Best: " << *min << '\n';
    std::cout << "Average: " << average << '\n';
    std::cout << "Average time: " << averageTime << '\n';

    return 0;
}