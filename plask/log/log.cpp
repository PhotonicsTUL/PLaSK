#include <cstdio>
using namespace std;

#include "log.h"

#ifndef _WIN32
#   include <unistd.h>
#endif

namespace plask {

#ifdef _WIN32
    struct StderrLogger {
        static std::string head(LogLevel level) {
            switch (level) {
                case LOG_ERROR:         return "ERROR         ";
                case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
                case LOG_WARNING:       return "WARNING       ";
                case LOG_INFO:          return "INFO          ";
                case LOG_RESULT:        return "RESULT        ";
                case LOG_DATA:          return "DATA          ";
                case LOG_DETAIL:        return "DETAIL        ";
                case LOG_ERROR_DETAIL:  return "ERROR DETAIL  ";
                case LOG_DEBUG:         return "DEBUG         ";
            }
        }

        static void writelog(LogLevel level, const std::string& msg) {
            fprintf(stderr, "%s: %s\n", head(level).c_str(), msg.c_str());
        }
    };
#else
    struct StderrLogger {
        static const bool tty;

        static std::string head(LogLevel level) {
            if (tty) {
                switch (level) {
                    // case LOG_CRITICAL_ERROR:return "\033[41;37;01mCRITICAL ERROR";
                    case LOG_CRITICAL_ERROR:return "\033[31;01mCRITICAL ERROR";
                    case LOG_ERROR:         return "\033[31;01mERROR         ";
                    case LOG_WARNING:       return "\033[31mWARNING       ";
                    case LOG_INFO:          return "\033[36mINFO          ";
                    case LOG_RESULT:        return "\033[32mRESULT        ";
                    case LOG_DATA:          return "\033[33mDATA          ";
                    case LOG_DETAIL:        return "\033[00mDETAIL        ";
                    case LOG_ERROR_DETAIL:  return "\033[31mERROR DETAIL  ";
                    case LOG_DEBUG:         return "\033[30;01;03mDEBUG         ";
                }
            } else {
                switch (level) {
                    case LOG_ERROR:         return "ERROR         ";
                    case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
                    case LOG_WARNING:       return "WARNING       ";
                    case LOG_INFO:          return "INFO          ";
                    case LOG_RESULT:        return "RESULT        ";
                    case LOG_DATA:          return "DATA          ";
                    case LOG_DETAIL:        return "DETAIL        ";
                    case LOG_ERROR_DETAIL:  return "ERROR DETAIL  ";
                    case LOG_DEBUG:         return "DEBUG         ";
                }
            }
        }

        static void writelog(LogLevel level, const std::string& msg) {
            fprintf(stderr, "%s: %s%s\n", head(level).c_str(), msg.c_str(), tty? "\033[0m" : "");
        }
    };
    const bool StderrLogger::tty = isatty(fileno(stderr));
#endif

void writelog(LogLevel level, const std::string& msg) {
    #pragma omp critical(writelog)
    StderrLogger::writelog(level, msg);
}

} // namespace plask
