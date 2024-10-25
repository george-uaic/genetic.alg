#include "CudaHillClimbing.cuh"
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <numbers>
#include <random>

// https://forums.developer.nvidia.com/t/intellisense-error-in-brand-new-cuda-project-in-vs2019/111921/8
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

__constant__ Config d_config;

__device__ double rastrigin(const double* x, int n)
{
    double sum = 10.0 * n;
    for (int i = 0; i < n; ++i)
    {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * std::numbers::pi * x[i]);
    }
    return sum;
}

__global__ void setupRandomStates(curandState* states, unsigned long long seed)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    seed ^= (idx << 31) - 1; // seed XOR-ed with idx * mersenne prime 2^31 - 1
    curand_init(seed, idx, 1000, &states[idx]);
}

// bitstrings:
// [Iter0 Dim0 Bits][Iter0 Dim1 Bits]...[Iter0 DimN Bits][Iter1 Dim0 Bits][Iter1 Dim1 Bits]...
__global__ void generateInitialBitstrings(curandState* states, bool* bitstrings)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    curandState localState = states[idx];
    int start_bit = idx * d_config.total_bits;

    for (int j = 0; j < d_config.total_bits; ++j)
    {
        bitstrings[start_bit + j] = curand_uniform(&localState) > 0.5f;
    }

    states[idx] = localState;
}

__device__ void convertBitstringToReal(const bool* bitstring, double* real_value)
{
    for (int dim = 0; dim < d_config.dimensions; ++dim)
    {
        unsigned long long decimal = 0;
        for (int bit = 0; bit < d_config.bits_per_dim; ++bit)
        {
            int bitIdx = dim * d_config.bits_per_dim + bit;
            decimal = (decimal << 1) | bitstring[bitIdx];
        }
        real_value[dim] = d_config.interval_lower + decimal * d_config.cached_value_for_conversion;
    }
}

// real_values:
// [Iter0 Dim0 Value] [Iter0 Dim1 Value] ... [Iter0 DimN Value] [Iter1 Dim0 Value] [Iter1 Dim1 Value]...
__global__ void convertToRealValues(const bool* bitstrings, double* real_values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);
}

__global__ void evaluateFitness(const double* real_values, double* fitness_values) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    fitness_values[idx] = rastrigin(&real_values[idx * d_config.dimensions], d_config.dimensions);
}

__global__ void hillClimbWorst(curandState* states, bool* bitstrings, double* real_values, double* fitness_values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    int start_bit = idx * d_config.total_bits;
    double current_fitness = fitness_values[idx];
    double worst_neighbor_fitness = current_fitness;
    int worst_neighbor_bit = -1;

    bool improved = true;
    while (improved)
    {
        improved = false;
        worst_neighbor_fitness = current_fitness;
        worst_neighbor_bit = -1;

        for (int i = 0; i < d_config.total_bits; ++i)
        {
            int bit_to_flip = start_bit + i;
            bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];

            convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);
            double new_fitness = rastrigin(&real_values[idx * d_config.dimensions], d_config.dimensions);

            if (new_fitness < current_fitness)
            {
                if (worst_neighbor_fitness == current_fitness || worst_neighbor_fitness < new_fitness)
                {
                    worst_neighbor_fitness = new_fitness;
                    worst_neighbor_bit = bit_to_flip;
                    improved = true;
                }
            }

            bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];
        }

        if (improved)
        {
            bitstrings[worst_neighbor_bit] = !bitstrings[worst_neighbor_bit];

            convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);

            current_fitness = worst_neighbor_fitness;
            fitness_values[idx] = current_fitness;
        }
    }
}

__global__ void hillClimbBest(curandState* states, bool* bitstrings, double* real_values, double* fitness_values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    int start_bit = idx * d_config.total_bits;
    double current_fitness = fitness_values[idx];
    double best_neighbor_fitness = current_fitness;
    int best_neighbor_bit = -1;

    bool improved = true;
    while (improved)
    {
        improved = false;
        best_neighbor_fitness = current_fitness;
        best_neighbor_bit = -1;

        for (int i = 0; i < d_config.total_bits; ++i)
        {
            int bit_to_flip = start_bit + i;
            bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];

            convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);
            double new_fitness = rastrigin(&real_values[idx * d_config.dimensions], d_config.dimensions);

            if (new_fitness < best_neighbor_fitness)
            {
                best_neighbor_fitness = new_fitness;
                best_neighbor_bit = bit_to_flip;
                improved = true;
            }

            bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];
        }

        if (improved)
        {
            bitstrings[best_neighbor_bit] = !bitstrings[best_neighbor_bit];

            convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);

            current_fitness = best_neighbor_fitness;
            fitness_values[idx] = current_fitness;
        }
    }
}

__global__ void hillClimbFirst(curandState* states, bool* bitstrings, double* real_values, double* fitness_values) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    int start_bit = idx * d_config.total_bits;
    double current_fitness = fitness_values[idx];

    for (int i = 0; i < d_config.total_bits; ++i)
    {
        int bit_to_flip = start_bit + i;
        bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];
    
        convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);
        double new_fitness = rastrigin(&real_values[idx * d_config.dimensions], d_config.dimensions);
        
        if (new_fitness < current_fitness)
        {
            current_fitness = new_fitness;
            fitness_values[idx] = new_fitness;
            i = 0;
        }
        else bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];
    }
}

__global__ void simulatedAnnealing(curandState* states, bool* bitstrings, double* real_values, double* fitness_values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_config.iterations) return;

    int start_bit = idx * d_config.total_bits;
    double current_fitness = fitness_values[idx];

    double cooling_rate = 0.98;
    double T0 = abs((40.35329019 * d_config.dimensions) / log(0.8));
    double T = T0;
    double temperature_treshold = T0 * 1e-8;
    int t = 0;
    int k = 0;
    int L = 25 * d_config.dimensions;

    while (k <= 5 && T > temperature_treshold && t < 75000)
    {
        int total_attempts = 0;
        int successful_attempts = 0;
        while (successful_attempts < L && total_attempts < 10 * L)
        {
            int neighbor = curand_uniform(&states[idx]) * d_config.total_bits;
            int bit_to_flip = start_bit + neighbor;
            bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];

            convertBitstringToReal(&bitstrings[idx * d_config.total_bits], &real_values[idx * d_config.dimensions]);
            double new_fitness = rastrigin(&real_values[idx * d_config.dimensions], d_config.dimensions);

            if (new_fitness < current_fitness)
            {
                current_fitness = new_fitness;
                fitness_values[idx] = new_fitness;
                successful_attempts++;
            }
            else if (curand_uniform(&states[idx]) < exp(-abs(new_fitness - current_fitness) / T))
            {
                current_fitness = new_fitness;
                fitness_values[idx] = new_fitness;
                successful_attempts++;
            }
            else bitstrings[bit_to_flip] = !bitstrings[bit_to_flip];

            total_attempts++;
        }
        t++;
        T = T0 * pow(cooling_rate, t);
        if (successful_attempts <= 2) k++;
        else k = 0;
    }
}

std::vector<double> launchHillClimb(const Config& config)
{
#ifndef __INTELLISENSE__
    cudaMemcpyToSymbol(d_config, &config, sizeof(Config));
#endif

    bool* d_bitstrings;
    double* d_real_values;
    double* d_fitness_values;
    curandState* d_rand_states;

    cudaMalloc(&d_bitstrings, sizeof(bool) * config.iterations * config.total_bits);
    cudaMalloc(&d_real_values, sizeof(double) * config.iterations * config.dimensions);
    cudaMalloc(&d_fitness_values, sizeof(double) * config.iterations);
    cudaMalloc(&d_rand_states, sizeof(curandState) * config.threads_per_block * config.blocks);

    setupRandomStates KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_rand_states, std::random_device{}());
    generateInitialBitstrings KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_rand_states, d_bitstrings);
    convertToRealValues KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_bitstrings, d_real_values);

    // evaluate initial fitness values
    evaluateFitness KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_real_values, d_fitness_values);

    if (config.improvement_strategy == ImprovementStrategy::FirstImprovement)
    {
        hillClimbFirst KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_rand_states, d_bitstrings, d_real_values, d_fitness_values);
    }
    else if (config.improvement_strategy == ImprovementStrategy::BestImprovement)
    {
        hillClimbBest KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_rand_states, d_bitstrings, d_real_values, d_fitness_values);
    }
    else if (config.improvement_strategy == ImprovementStrategy::WorstImprovement)
    {
        hillClimbWorst KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_rand_states, d_bitstrings, d_real_values, d_fitness_values);
    }
    else if(config.improvement_strategy == ImprovementStrategy::SimulatedAnnealing)
    {
        simulatedAnnealing KERNEL_ARGS2(config.blocks, config.threads_per_block) (d_rand_states, d_bitstrings, d_real_values, d_fitness_values);
    }

    // copy results back to host
    std::vector<double> host_fitness(config.iterations);
    cudaMemcpy(host_fitness.data(), d_fitness_values, config.iterations * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_bitstrings);
    cudaFree(d_real_values);
    cudaFree(d_fitness_values);
    cudaFree(d_rand_states);

    return host_fitness;
}