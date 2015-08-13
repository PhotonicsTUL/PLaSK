#include <cstdio>
using namespace std;

#include "log.h"

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   include <windows.h>
#   define BOOST_USE_WINDOWS_H
#else
#   include <unistd.h>
#endif

namespace plask {

#ifdef NDEBUG
PLASK_API LogLevel maxLoglevel = LOG_DETAIL;
#else
PLASK_API LogLevel maxLoglevel = LOG_DEBUG;
#endif

PLASK_API bool forcedLoglevel = false;


struct StderrLogger: public plask::Logger {

#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    void setcolor(unsigned short fg);
    unsigned short previous_color;
#   endif

    const char* head(plask::LogLevel level);

    virtual void writelog(plask::LogLevel level, const std::string& msg);

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
        HANDLE handle = GetStdHandle((dest==DEST_STDERR)?STD_ERROR_HANDLE:STD_OUTPUT_HANDLE);
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
            case LOG_CRITICAL_ERROR:return ANSI_BRIGHT_RED  "CRITICAL ERROR";
            case LOG_ERROR:         return ANSI_BRIGHT_RED  "ERROR         ";
            case LOG_WARNING:       return ANSI_BROWN       "WARNING       ";
            case LOG_INFO:          return ANSI_BRIGHT_BLUE "INFO          ";
            case LOG_RESULT:        return ANSI_GREEN       "RESULT        ";
            case LOG_DATA:          return ANSI_CYAN        "DATA          ";
            case LOG_DETAIL:        return ANSI_DEFAULT     "DETAIL        ";
            case LOG_ERROR_DETAIL:  return ANSI_RED         "ERROR DETAIL  ";
            case LOG_DEBUG:         return ANSI_GRAY        "DEBUG         ";
        }
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    else if (color == StderrLogger::COLOR_WINDOWS)
        switch (level) {
            case LOG_ERROR:         setcolor(COL_BRIGHT_RED); return "ERROR         ";
            case LOG_CRITICAL_ERROR:setcolor(COL_BRIGHT_RED); return "CRITICAL ERROR";
            case LOG_WARNING:       setcolor(COL_BROWN); return "WARNING       ";
            case LOG_INFO:          setcolor(COL_BRIGHT_CYAN); return "INFO          ";
            case LOG_RESULT:        setcolor(COL_GREEN); return "RESULT        ";
            case LOG_DATA:          setcolor(COL_CYAN); return "DATA          ";
            case LOG_DETAIL:        return "DETAIL        ";
            case LOG_ERROR_DETAIL:  setcolor(COL_RED); return "ERROR DETAIL  ";
            case LOG_DEBUG:         setcolor(COL_GRAY); return "DEBUG         ";
        }
#endif
    else
        switch (level) {
            case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
            case LOG_ERROR:         return "ERROR         ";
            case LOG_WARNING:       return "WARNING       ";
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
    // PyFrameObject* frame = PyEval_GetFrame();
    // if (frame)
    //     pyinfo = format("%2%:%1%: ", PyFrame_GetLineNumber(frame), PyString_AsString(frame->f_code->co_filename));
    if (color == COLOR_ANSI) {
        #pragma omp critical
        fprintf(stderr, "%s: %s" ANSI_DEFAULT "\n", head(level), msg.c_str());
    #if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    } else if (color == COLOR_WINDOWS) {
        #pragma omp critical
        fprintf(stderr, "%s: %s\n", head(level), msg.c_str());
        SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), previous_color);
    #endif
    } else {
        #pragma omp critical
        fprintf(stderr, "%s: %s\n", head(level), msg.c_str());
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
