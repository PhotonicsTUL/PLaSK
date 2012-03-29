#ifndef PLASK__LOG_LOG_H
#define PLASK__LOG_LOG_H

namespace plask {

enum LogLevel {
    LOG_CRITICAL_ERROR, // these errors are handled by exceptions, but this is for completeness
    LOG_ERROR,          // non-critical errors (e.g. failed convergence etc.), user can decide if the program should continue
    LOG_WARNING,        // warning
    LOG_INFO,           // basic log level, gives general information on the program flow
    LOG_RESULT,         // single value results (also provided with provider mechanism) for easy tracking
    LOG_DATA,           // intermediate data/results, mainly for presentation in live plots
    LOG_DETAIL,         // less important details on computations (i.e. recomputations of Jacobian in Broyden method)
    LOG_DEBUG           // pretty much everything
};

template<typename ...Args>
inline void log(LogLevel level, Args&&... params) {
    // format(std::forward<Args>(params)...);
}

}   // namespace plask


#endif // PLASK__LOG_LOG_H
