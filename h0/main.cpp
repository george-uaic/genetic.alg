#include <iostream>
#include <iomanip> 

#include <random>
#include <ranges>

#include "Functions.h"
#include "Settings.h"

std::mt19937 mt{ std::random_device{}() };
std::bernoulli_distribution dist(0.5);

using bitstring = std::vector<bool>;


static std::vector<double> Convert(const bitstring& bitstr)
{
    auto toDecimal = [](int acc, bool bit) { return (acc << 1) | (int)bit; };

    return bitstr
    | std::views::chunk(Settings::bits)
    | std::views::transform([&toDecimal](auto&& chunk)
      {
          int decimal{ std::accumulate(chunk.begin(), chunk.end(), 0, toDecimal) };
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

static double IteratedHillClimbing()
{
    int totalBits{ Settings::bits * Settings::funcDef.dimensions };
    
    constexpr double inf{ std::numeric_limits<double>::infinity() };
    double best{ Settings::ext == Extrema::Min ? inf : -inf };
   
    for (int t{ 0 }; t < Settings::iterCount; t++)
    {
        bitstring vc(totalBits);
        std::ranges::generate_n(vc.begin(), totalBits, []() { return dist(mt); });

        while (Improve(vc, Neighborhood(vc)));

        double candidate = Settings::funcDef.function(Convert(vc));
        if (Better(candidate, best) > 0) best = candidate;

        if (t % 100 == 0) std::cout << t << ": current best = " << best << '\n';
    }

    return best;
}

int main()
{
    FunctionDef funcDef{ -5.12, 5.12, 2, Rastrigin };
    //FunctionDef funcDef{ 0, std::numbers::pi, 10, Michalewicz };
    Settings::Initialize(funcDef, Extrema::Min, Improvement::BestImprovement, 10000, 5);

    std::cout << std::fixed << std::setprecision(Settings::precision) << IteratedHillClimbing();
}