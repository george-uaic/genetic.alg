#include "Clock.h"

Time::Time(LargeInt time) : time(time)
{
}

Time::LargeInt Time::asMicroseconds() const
{
    return time;
}

Time::LargeInt Time::asMilliseconds() const
{
    return time / 1000;
}

float Time::asSeconds() const
{
    return time / 1000000.f;
}

Time Time::Seconds(float seconds)
{
    return (LargeInt)(seconds * 1000000);
}

Time Time::Milliseconds(LargeInt milliseconds)
{
    return milliseconds * 1000;
}

Clock::Clock(bool start) : clockPaused(!start)
{
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&beginTime);
}

Time Clock::GetElapsedTime()
{
    if (!clockPaused)
    {
        QueryPerformanceCounter(&endTime);
        elapsedTime += ((endTime.QuadPart - beginTime.QuadPart) * 1000000) / frequency.QuadPart;
        QueryPerformanceCounter(&beginTime);
    }

    return Time(elapsedTime);
}

void Clock::Reset()
{
    elapsedTime = 0;
}

void Clock::Pause()
{
    clockPaused = true;
}

void Clock::Restart()
{
    if (clockPaused)
    {
        QueryPerformanceCounter(&beginTime);
        clockPaused = false;
    }
}

bool Clock::IsPaused() const
{
    return clockPaused;
}