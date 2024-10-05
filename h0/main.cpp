#include <iostream>
#include <iomanip> 

#include <random>
#include <ranges>

#include <future>

#include "Functions.h"
#include "Settings.h"
#include "Clock.h"

std::mt19937 mt{ std::random_device{}() };
std::bernoulli_distribution dist(0.5);

using bitstring = std::vector<bool>;


static std::vector<double> Convert(const bitstring& bitstr)
{
    auto toDecimal = [](unsigned long long acc, bool bit) { return (acc << 1) | (unsigned long long)bit; };

    return bitstr
    | std::views::chunk(Settings::bits)
    | std::views::transform([&toDecimal](auto&& chunk)
      {
          unsigned long long decimal{ std::accumulate(chunk.begin(), chunk.end(), (unsigned long long)0, toDecimal) };
          double a{ Settings::funcDef.interval_lower };
          double b{ Settings::funcDef.interval_upper };
          return a + decimal * (b - a) / (pow(2, Settings::bits) - 1);
      })
    | std::ranges::to<std::vector<double>>();
}

// returns how much better
static double Better(double candidate, double target)
{
    return Settings::ext == Extrema::Min ? target - candidate : candidate - target;
}

static std::vector<bitstring> Neighborhood(const bitstring& vc)
{
    std::vector<bitstring> neighbors(vc.size(), vc);

    for (std::size_t i{ 0 }; i < vc.size(); i++)
    {
        neighbors[i][i] = !vc[i];
    }

    return neighbors;
}

// returns true if improved, false otherwise
static bool Improve(bitstring& current, const std::vector<bitstring>& neighbors)
{
    double currentOutput{ Settings::funcDef.function(Convert(current)) };
    double bestImprovement{ 0.0 };
    const bitstring* bestNeighbor{ nullptr };

    for (const bitstring& neighbor : neighbors)
    {
        double neighborOutput{ Settings::funcDef.function(Convert(neighbor)) };
        double improvement{ Better(neighborOutput, currentOutput) };

        if (Settings::improv == Improvement::FirstImprovement && improvement > 0.0)
        {
            current = neighbor;
            return true;
        }

        if (improvement > bestImprovement)
        {
            bestImprovement = improvement;
            bestNeighbor = &neighbor;
        }
    }

    if (bestImprovement > 0.0)
    {
        current = *bestNeighbor;
        return true;
    }
    else return false;
}

static double IteratedHillClimbing(int iterations, int thread)
{
    int totalBits{ Settings::bits * Settings::funcDef.dimensions };
    
    constexpr double inf{ std::numeric_limits<double>::infinity() };
    double localBest{ Settings::ext == Extrema::Min ? inf : -inf };
   
    for (int t{ 0 }; t < iterations; t++)
    {
        bitstring vc(totalBits);
        std::ranges::generate_n(vc.begin(), totalBits, []() { return dist(mt); });

        while (Improve(vc, Neighborhood(vc)));

        double candidate = Settings::funcDef.function(Convert(vc));
        if (Better(candidate, localBest) > 0) localBest = candidate;

        if (t % 100 == 0)
        {
            std::cout << "Thread: " << thread << " Iteration: " << t;
            std::cout << " Local Best = " << localBest << '\n';
        }
    }

    return localBest;
}



int main()
{
    FunctionDef funcDef{ -5.12, 5.12, 10, Rastrigin };
    //FunctionDef funcDef{ 0, std::numbers::pi, 10, Michalewicz };
    Settings::Initialize(funcDef, Extrema::Max, Improvement::BestImprovement, 10000, 5);
    std::cout << std::fixed << std::setprecision(Settings::precision);

    Clock clock;

    int threadCount = std::thread::hardware_concurrency();
    int iterPerThread = Settings::iterCount / threadCount;
    int remainder = Settings::iterCount % threadCount;
    
    std::vector<std::future<double>> futures;
    
    for (int i{ 1 }; i <= threadCount; i++)
    {
        int bonus = i <= remainder ? 1 : 0;
        futures.push_back(std::async(std::launch::async, IteratedHillClimbing, iterPerThread + bonus, i));
    }
    
    constexpr double inf{ std::numeric_limits<double>::infinity() };
    double globalBest{ Settings::ext == Extrema::Min ? inf : -inf };
    
    for (auto& f : futures)
    {
        double candidate{ f.get() };
        if (Better(candidate, globalBest) > 0) globalBest = candidate;
    }
    
    std::cout << "Global Best: " << globalBest << '\n';
    
    std::cout << clock.GetElapsedTime().asSeconds() << " seconds";
}