#include "log.h"

namespace plask {

static std::string logLevelHead(LogLevel level) {
    switch (level) {
        case LOG_CRITICAL_ERROR:return "\033[41;37;01mCRITICAL ERROR";
        case LOG_ERROR:         return "\033[31;01mERROR         ";
        case LOG_WARNING:       return "\033[31mWARNING       ";
        case LOG_INFO:          return "\033[34mINFO          ";
        case LOG_RESULT:        return "\033[32mRESULT        ";
        case LOG_DATA:          return "\033[33mDATA          ";
        case LOG_DETAIL:        return "\033[30mDETAIL        ";
        case LOG_DEBUG:         return "\033[37mDEBUG         ";
        // case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
        // case LOG_ERROR:         return "ERROR         ";
        // case LOG_WARNING:       return "WARNING       ";
        // case LOG_INFO:          return "INFO          ";
        // case LOG_RESULT:        return "RESULT        ";
        // case LOG_DATA:          return "DATA          ";
        // case LOG_DETAIL:        return "DETAIL        ";
        // case LOG_DEBUG:         return "DEBUG         ";
    }
    return "";
}

void writelog(LogLevel level, const std::string& msg) {
    std::cerr << logLevelHead(level) << ": " <<  msg << "\033[0m" "\n";
}

} // namspace plask