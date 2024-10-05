#include "Settings.h"

void Settings::Initialize(const FunctionDef& _funcDef, Extrema _ext, Improvement _improv, int _iterCount, int _precision)
{
    ext = _ext;
    improv = _improv;
    iterCount = _iterCount;
    precision = _precision;
    setFunction(_funcDef);
}

void Settings::setFunction(const FunctionDef& _funcDef)
{
    double N = (_funcDef.interval_upper - _funcDef.interval_lower) * pow(10, precision);
    bits = (int)ceil(log2(N));

    funcDef = _funcDef;
}

FunctionDef Settings::funcDef;
Extrema Settings::ext;
Improvement Settings::improv;
int Settings::iterCount;
int Settings::precision;
int Settings::bits;