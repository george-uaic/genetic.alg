#pragma once
#include <functional>
#include <vector>
#include <string>

enum class Extrema
{
    Min, Max
};

enum class Improvement
{
    FirstImprovement,
    BestImprovement
};

struct FunctionDef
{
    double interval_lower = 0;
    double interval_upper = 0;
    int dimensions = 0;
    std::function<double(std::vector<double>)> function;
};

struct SettingsStruct
{
    FunctionDef funcDef;
    Extrema ext;
    Improvement improv;
    std::string id;
    int iterCount = 10000;
    int precision = 5;
};

class Settings
{
public:

    Settings() = delete;

    static void Initialize(const FunctionDef& _funcDef, Extrema _ext, Improvement _improv, int _iterCount = 10000, int _precision = 5);
    static void setFunction(const FunctionDef& _funcDef);

    static FunctionDef funcDef;
    static Extrema ext;
    static Improvement improv;
    static int iterCount;
    static int precision;
    static int bits;
};