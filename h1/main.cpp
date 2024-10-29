#include "CudaHillClimbing.cuh"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <numbers>

int main() 
{
    Config config{};
    config.function = BenchmarkFunction::Rastrigin;
    config.improvement_strategy = ImprovementStrategy::SimulatedAnnealing;

    switch (config.function)
    {
        case BenchmarkFunction::Rastrigin:
        {
            config.interval_lower = -5.12;
            config.interval_upper = 5.12;
            break;
        }
        case BenchmarkFunction::Schwefel:
        {
            config.interval_lower = -500;
            config.interval_upper = 500;
            break;
        }
        case BenchmarkFunction::Sphere:
        {
            config.interval_lower = -5.12;
            config.interval_upper = 5.12;
            break;
        }
        case BenchmarkFunction::Michalewicz:
        {
            config.interval_lower = 0;
            config.interval_upper = std::numbers::pi;
            break;
        }
    }

    config.precision = 5;
    config.dimensions = 100;
    double segments = (config.interval_upper - config.interval_lower) * pow(10, config.precision);
    config.bits_per_dim = (int)ceil(log2(segments));
    config.total_bits = config.dimensions * config.bits_per_dim;
    config.cached_value_for_conversion = (config.interval_upper - config.interval_lower) / ((1ull << config.bits_per_dim) - 1);

    config.iterations = 20000;
    config.threads_per_block = 32;
    config.blocks = (config.iterations + config.threads_per_block - 1) / config.threads_per_block;

    std::cout << std::fixed << std::setprecision(config.precision);

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

    std::ofstream out("rastrigin_100D_simulated_annealing.txt");
    out << std::fixed << std::setprecision(config.precision);
    for (int i = 0; i < results.size(); i++)
    {
        out << results[i] << ' ';
    }
    out << "\nAverage: " << average << "\nBest: " << *min << "\nAverage Time: " << averageTime << "ms";

    return 0;
}