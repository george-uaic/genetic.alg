#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class Time
{
    using LargeInt = long long;
    LargeInt time;

public:
    Time(LargeInt time);
    LargeInt asMicroseconds() const;
    LargeInt asMilliseconds() const;
    double asSeconds() const;

    static Time Seconds(double seconds);
    static Time Milliseconds(LargeInt milliseconds);
};

class Clock
{
    LARGE_INTEGER beginTime{};
    LARGE_INTEGER endTime{};
    LARGE_INTEGER frequency{};
    long long elapsedTime{};
    bool clockPaused;

public:
    Clock(bool start = true);
    Time GetElapsedTime();
    void Reset();
    void Pause();
    void Restart();
    bool IsPaused() const;
};