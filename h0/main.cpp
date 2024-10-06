#include <fstream>
#include <iomanip> 
#include <random>
#include <ranges>
#include <algorithm>
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
    }

    return localBest;
}


int main()
{
    FunctionDef rastrigin2{ -5.12, 5.12, 2, Rastrigin };
    FunctionDef rastrigin10{ -5.12, 5.12, 10, Rastrigin };
    FunctionDef michalewicz2{ 0, std::numbers::pi, 2, Michalewicz };
    FunctionDef michalewicz10{ 0, std::numbers::pi, 10, Michalewicz };
    FunctionDef sphere2{ -5.12, 5.12, 2, Sphere };
    FunctionDef sphere10{ -5.12, 5.12, 10, Sphere };
    FunctionDef griewank2{ -600, 600, 2, Griewank };
    FunctionDef griewank10{ -600, 600, 10, Griewank };

    std::vector<SettingsStruct> settings;
    settings.push_back({ rastrigin2, Extrema::Min, Improvement::BestImprovement, "Rastrigin_2D_Min_BI" });
    settings.push_back({ rastrigin10, Extrema::Min, Improvement::BestImprovement, "Rastrigin_10D_Min_BI" });
    settings.push_back({ rastrigin2, Extrema::Max, Improvement::BestImprovement, "Rastrigin_2D_Max_BI" });
    settings.push_back({ rastrigin10, Extrema::Max, Improvement::BestImprovement, "Rastrigin_10D_Max_BI" });
    settings.push_back({ rastrigin2, Extrema::Min, Improvement::FirstImprovement, "Rastrigin_2D_Min_FI" });
    settings.push_back({ rastrigin10, Extrema::Min, Improvement::FirstImprovement, "Rastrigin_10D_Min_FI" });
    settings.push_back({ rastrigin2, Extrema::Max, Improvement::FirstImprovement, "Rastrigin_2D_Max_FI" });
    settings.push_back({ rastrigin10, Extrema::Max, Improvement::FirstImprovement, "Rastrigin_10D_Max_FI" });

    settings.push_back({ michalewicz2, Extrema::Min, Improvement::BestImprovement, "Michalewicz_2D_Min_BI" });
    settings.push_back({ michalewicz10, Extrema::Min, Improvement::BestImprovement, "Michalewicz_10D_Min_BI" });
    settings.push_back({ michalewicz2, Extrema::Max, Improvement::BestImprovement, "Michalewicz_2D_Max_BI" });
    settings.push_back({ michalewicz10, Extrema::Max, Improvement::BestImprovement, "Michalewicz_10D_Max_BI" });
    settings.push_back({ michalewicz2, Extrema::Min, Improvement::FirstImprovement, "Michalewicz_2D_Min_FI" });
    settings.push_back({ michalewicz10, Extrema::Min, Improvement::FirstImprovement, "Michalewicz_10D_Min_FI" });
    settings.push_back({ michalewicz2, Extrema::Max, Improvement::FirstImprovement, "Michalewicz_2D_Max_FI" });
    settings.push_back({ michalewicz10, Extrema::Max, Improvement::FirstImprovement, "Michalewicz_10D_Max_FI" });

    settings.push_back({ sphere2, Extrema::Min, Improvement::BestImprovement, "Sphere_2D_Min_BI" });
    settings.push_back({ sphere10, Extrema::Min, Improvement::BestImprovement, "Sphere_10D_Min_BI" });
    settings.push_back({ sphere2, Extrema::Max, Improvement::BestImprovement, "Sphere_2D_Max_BI" });
    settings.push_back({ sphere10, Extrema::Max, Improvement::BestImprovement, "Sphere_10D_Max_BI" });
    settings.push_back({ sphere2, Extrema::Min, Improvement::FirstImprovement, "Sphere_2D_Min_FI" });
    settings.push_back({ sphere10, Extrema::Min, Improvement::FirstImprovement, "Sphere_10D_Min_FI" });
    settings.push_back({ sphere2, Extrema::Max, Improvement::FirstImprovement, "Sphere_2D_Max_FI" });
    settings.push_back({ sphere10, Extrema::Max, Improvement::FirstImprovement, "Sphere_10D_Max_FI" });

    settings.push_back({ griewank2, Extrema::Min, Improvement::BestImprovement, "Griewank_2D_Min_BI" });
    settings.push_back({ griewank10, Extrema::Min, Improvement::BestImprovement, "Griewank_10D_Min_BI" });
    settings.push_back({ griewank2, Extrema::Max, Improvement::BestImprovement, "Griewank_2D_Max_BI" });
    settings.push_back({ griewank10, Extrema::Max, Improvement::BestImprovement, "Griewank_10D_Max_BI" });
    settings.push_back({ griewank2, Extrema::Min, Improvement::FirstImprovement, "Griewank_2D_Min_FI" });
    settings.push_back({ griewank10, Extrema::Min, Improvement::FirstImprovement, "Griewank_10D_Min_FI" });
    settings.push_back({ griewank2, Extrema::Max, Improvement::FirstImprovement, "Griewank_2D_Max_FI" });
    settings.push_back({ griewank10, Extrema::Max, Improvement::FirstImprovement, "Griewank_10D_Max_FI" });

    
    int threadCount = std::thread::hardware_concurrency();
    Clock clock;

    for (const SettingsStruct& setting : settings)
    {
        Settings::Initialize(setting.funcDef, setting.ext, setting.improv, setting.iterCount, setting.precision);
        
        int iterPerThread = Settings::iterCount / threadCount;
        int remainder = Settings::iterCount % threadCount;

        std::vector<double> executionTime;
        std::vector<double> results;

        int samples{ 30 };
        for (int i{ 1 }; i <= samples; i++)
        {
            clock.Restart();

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

            results.push_back(globalBest);
            executionTime.push_back(clock.GetElapsedTime().asSeconds());
            clock.Reset();
        }

        std::ofstream out(setting.id + ".txt");
        out << std::fixed << std::setprecision(Settings::precision);

        out << "Results: ";
        for (int i{ 0 }; i < results.size(); i++) out << results[i] << ' ';
        out << '\n';

        if (Settings::ext == Extrema::Max)
        {
            out << "Best: " << *std::max_element(results.begin(), results.end()) << '\n';
            out << "Average: " << std::accumulate(results.begin(), results.end(), 0.0) / (double) results.size() << '\n';
            out << "Worst: " << *std::min_element(results.begin(), results.end()) << '\n';
        }
        else if (Settings::ext == Extrema::Min)
        {
            out << "Best: " << *std::min_element(results.begin(), results.end()) << '\n';
            out << "Average: " << std::accumulate(results.begin(), results.end(), 0.0) / (double)results.size() << '\n';
            out << "Worst: " << *std::max_element(results.begin(), results.end()) << '\n';
        }

        out << "\nExecution Time: ";
        for (int i{ 0 }; i < executionTime.size(); i++) out << executionTime[i] << ' ';
        out << '\n';

        out << "Best: " << *std::min_element(executionTime.begin(), executionTime.end()) << '\n';
        out << "Average: " << std::accumulate(executionTime.begin(), executionTime.end(), 0.0) / (double)executionTime.size() << '\n';
        out << "Worst: " << *std::max_element(executionTime.begin(), executionTime.end()) << '\n';
        out.close();
    }
}