#ifndef PLASK__LOG_LOG_H
#define PLASK__LOG_LOG_H

#include <iostream>
#include <string>

#include "../utils/format.h"

namespace plask {

enum LogLevel {
    LOG_CRITICAL_ERROR, ///< Exceptions and errors unconditionally interrupting program flow
    LOG_ERROR,          ///< Non-critical errors (e.g. failed convergence etc.), user can decide if the program should continue
    LOG_WARNING,        ///< Warning
    LOG_INFO,           ///< Basic log level, gives general information on the program flow
    LOG_RESULT,         ///< Single value results (also provided with provider mechanism) for easy tracking
    LOG_DATA,           ///< Intermediate data/results, mainly for presentation in live plots
    LOG_DETAIL,         ///< Less important details on computations (i.e. recomputations of Jacobian in Broyden method)
    LOG_ERROR_DETAIL,   ///< Details of an error (e.g. stack trace)
    LOG_DEBUG           ///< Pretty much everything
};


void writelog(LogLevel level, const std::string& msg);

/**
 * Log a message
 * \param level log level to log
 * \param msg log message
 * \param params parameters passed to format
 **/
template<typename... Args>
inline void writelog(LogLevel level, const std::string& msg, Args&&... params) {
    writelog(level, format(msg, std::forward<Args>(params)...));
}

}   // namespace plask


#endif // PLASK__LOG_LOG_H
