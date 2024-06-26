/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include <cstdio>

#include "log.hpp"

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   include <plask/utils/minimal_windows.h>
#else
#   include <unistd.h>
#endif

#ifdef OPENMP_FOUND
#   include "../parallel.hpp"
#endif

namespace plask {

#ifdef NDEBUG
PLASK_API LogLevel maxLoglevel = LOG_DETAIL;
#else
PLASK_API LogLevel maxLoglevel = LOG_DEBUG;
#endif

PLASK_API bool forcedLoglevel = false;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    PLASK_API std::string host_name() {
        char name[1024];
		DWORD size = sizeof(name);
        GetComputerNameEx(ComputerNameDnsHostname, name, &size);
        return std::string(name);
    }
#else
    PLASK_API std::string host_name() {
        char name[1024];
        ::gethostname(name, sizeof(name));
        return std::string(name);
    }
#endif

Logger::Logger(): silent(false), color(
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        Logger::COLOR_WINDOWS
#   else
        isatty(fileno(stderr))? Logger::COLOR_ANSI : Logger::COLOR_NONE
#   endif
    ) {
    if (const char* env = std::getenv("OMPI_COMM_WORLD_RANK"))
        prefix = std::string(env) + " : ";
    else if (const char* env = std::getenv("PMI_RANK"))
        prefix = std::string(env) + " : ";
    else if (const char* env = std::getenv("SLURM_PROCID"))
        prefix = std::string(env) + " : ";
    else if (const char* env = std::getenv("PBS_VNODENUM"))
        prefix = std::string(env) + " : ";
    else
        prefix = "";
}

struct StderrLogger: public plask::Logger {

#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    void setcolor(unsigned short fg);
    unsigned short previous_color;
#   endif

    const char* head(plask::LogLevel level);

    void writelog(plask::LogLevel level, const std::string& msg) override;

};

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

#   define COL_BLACK 0
#   define COL_BLUE 1
#   define COL_GREEN 2
#   define COL_CYAN 3
#   define COL_RED 4
#   define COL_MAGENTA 5
#   define COL_BROWN 6
#   define COL_WHITE 7
#   define COL_GRAY 8
#   define COL_BRIGHT_BLUE 9
#   define COL_BRIGHT_GREEN 10
#   define COL_BRIGHT_CYAN 11
#   define COL_BRIGHT_RED 12
#   define COL_BRIGHT_MAGENTA 13
#   define COL_YELLOW 14
#   define COL_BRIGHT_WHITE 15

    inline void StderrLogger::setcolor(unsigned short fg) {
        HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(handle, &csbi);
        previous_color = csbi.wAttributes;
        SetConsoleTextAttribute(handle, (csbi.wAttributes & 0xF0) | fg);
    }

#else

#endif

#define ANSI_DEFAULT "\033[00m"
#define ANSI_BLACK   "\033[30m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_BROWN  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_WHITE   "\033[37m"
#define ANSI_GRAY   "\033[30;01m"
#define ANSI_BRIGHT_RED     "\033[31;01m"
#define ANSI_BRIGHT_GREEN   "\033[32;01m"
#define ANSI_YELLOW  "\033[33;01m"
#define ANSI_BRIGHT_BLUE    "\033[34;01m"
#define ANSI_BRIGHT_MAGENTA "\033[35;01m"
#define ANSI_BRIGHT_CYAN    "\033[36;01m"
#define ANSI_BRIGHT_WHITE   "\033[37;01m"
const char* StderrLogger::head(LogLevel level) {
    if (color == StderrLogger::COLOR_ANSI)
        switch (level) {
            case LOG_CRITICAL_ERROR:return ANSI_BRIGHT_RED     "CRITICAL ERROR";
            case LOG_ERROR:         return ANSI_BRIGHT_RED     "ERROR         ";
            case LOG_WARNING:       return ANSI_BROWN          "WARNING       ";
            case LOG_IMPORTANT:     return ANSI_BRIGHT_MAGENTA "IMPORTANT     ";
            case LOG_INFO:          return ANSI_BRIGHT_BLUE    "INFO          ";
            case LOG_RESULT:        return ANSI_GREEN          "RESULT        ";
            case LOG_DATA:          return ANSI_CYAN           "DATA          ";
            case LOG_DETAIL:        return ANSI_DEFAULT        "DETAIL        ";
            case LOG_ERROR_DETAIL:  return ANSI_RED            "ERROR DETAIL  ";
            case LOG_DEBUG:         return ANSI_GRAY           "DEBUG         ";
        }
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    else if (color == StderrLogger::COLOR_WINDOWS)
        switch (level) {
            case LOG_ERROR:         setcolor(COL_BRIGHT_RED);     return "ERROR         ";
            case LOG_CRITICAL_ERROR:setcolor(COL_BRIGHT_RED);     return "CRITICAL ERROR";
            case LOG_WARNING:       setcolor(COL_BROWN);          return "WARNING       ";
            case LOG_IMPORTANT:     setcolor(COL_BRIGHT_MAGENTA); return "IMPORTANT     ";
            case LOG_INFO:          setcolor(COL_BRIGHT_CYAN);    return "INFO          ";
            case LOG_RESULT:        setcolor(COL_GREEN);          return "RESULT        ";
            case LOG_DATA:          setcolor(COL_CYAN);           return "DATA          ";
            case LOG_DETAIL:                                      return "DETAIL        ";
            case LOG_ERROR_DETAIL:  setcolor(COL_RED);            return "ERROR DETAIL  ";
            case LOG_DEBUG:         setcolor(COL_GRAY);           return "DEBUG         ";
        }
#endif
    else
        switch (level) {
            case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
            case LOG_ERROR:         return "ERROR         ";
            case LOG_WARNING:       return "WARNING       ";
            case LOG_IMPORTANT:     return "IMPORTANT     ";
            case LOG_INFO:          return "INFO          ";
            case LOG_RESULT:        return "RESULT        ";
            case LOG_DATA:          return "DATA          ";
            case LOG_DETAIL:        return "DETAIL        ";
            case LOG_ERROR_DETAIL:  return "ERROR DETAIL  ";
            case LOG_DEBUG:         return "DEBUG         ";
        }
    return "UNSPECIFIED   "; // mostly to silence compiler warning than to use in the real life
}

void StderrLogger::writelog(LogLevel level, const std::string& msg) {
#ifdef OPENMP_FOUND
    static OmpSingleLock loglock;
    OmpLockGuard guard(loglock);
#endif

    static LogLevel prev_level; static std::string prev_msg;
    if (level == prev_level && msg == prev_msg) return;
    prev_level = level; prev_msg = msg;

    if (color == COLOR_ANSI) {
        fprintf(stderr, "%s: %s%s" ANSI_DEFAULT "\n", head(level), prefix.c_str(), msg.c_str());
    #if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    } else if (color == COLOR_WINDOWS) {
        fprintf(stderr, "%s: %s%s\n", head(level), prefix.c_str(), msg.c_str());
        SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), previous_color);
    #endif
    } else {
        fprintf(stderr, "%s: %s%s\n", head(level), prefix.c_str(), msg.c_str());
    }
}

PLASK_API shared_ptr<Logger> default_logger;

NoLogging::NoLogging(): old_state(default_logger->silent) {}


NoLogging::NoLogging(bool silent): old_state(default_logger->silent) {
    default_logger->silent = silent;
}

NoLogging::~NoLogging() {
    default_logger->silent = old_state;
}

/// Turn off logging in started without it
void NoLogging::set(bool silent) {
    default_logger->silent = silent;
}

/// Create default logger
PLASK_API void createDefaultLogger() {
    default_logger = shared_ptr<Logger>(new StderrLogger());
}

} // namespace plask
