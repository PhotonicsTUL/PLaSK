#ifndef PLASK__LOG_LOG_H
#define PLASK__LOG_LOG_H

#include <iostream>
#include <string>

#include "../memory.h"
#include "../utils/format.h"

namespace plask {

PLASK_API std::string host_name();
    
enum LogLevel {
    LOG_CRITICAL_ERROR = 0, ///< Exceptions and errors unconditionally interrupting program flow
    LOG_ERROR = 1,          ///< Non-critical errors (e.g. failed convergence etc.), user can decide if the program should continue
    LOG_ERROR_DETAIL = 2,   ///< Details of an error (e.g. stack trace)
    LOG_WARNING = 3,        ///< Warning
    LOG_INFO = 4,           ///< Basic log level, gives general information on the program flow
    LOG_RESULT = 5,         ///< Single value results (also provided with provider mechanism) for easy tracking
    LOG_DATA = 6,           ///< Intermediate data/results, mainly for presentation in live plots
    LOG_DETAIL = 7,         ///< Less important details on computations (i.e. recomputations of Jacobian in Broyden method)
    LOG_DEBUG = 8           ///< Pretty much everything
};

/// Maximum log level
PLASK_API extern LogLevel maxLoglevel;
PLASK_API extern bool forcedLoglevel;

/**
 * Logger switch.
 * Creating objects of these class temporarily turns off logging.
 */
class PLASK_API NoLogging {
    bool old_state;

  public:
    NoLogging();

    NoLogging(bool silent);

    ~NoLogging();

    /// Set logging state
    void set(bool silent);

    /// Turn off logging
    void silence() { set(true); }
};

/**
 * Abstract class that is base for all loggers
 */
class PLASK_API Logger {

    /// Flag indicating temporarily turned of logging
    bool silent;

    friend class NoLogging;

    friend void writelog(LogLevel level, const std::string& msg);

    template<typename... Args>
    friend void writelog(LogLevel level, const std::string& msg, Args&&... params);

  protected:

    /// Prefix to add to every log line
    std::string prefix;

  public:

    enum ColorMode {
        COLOR_NONE,
        COLOR_ANSI
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        , COLOR_WINDOWS
#endif
    };

    /// Log coloring mode
    ColorMode color;

    /// Get prefix
    std::string getPrefix() const { 
        size_t len = prefix.length();
        if (len) return prefix.substr(0, len-3);
        else return prefix;
    }

    /// Set prefix, parsing special fields
    virtual void setPrefix(const std::string& value);

    Logger(): silent(false), color(
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        Logger::COLOR_WINDOWS
#else
        isatty(fileno(stderr))? Logger::COLOR_ANSI : Logger::COLOR_NONE
#endif
    ) {}

    virtual ~Logger() {}

    /**
     * Log a message
     * \param level log level to log
     * \param msg log message
     */
    virtual void writelog(LogLevel level, const std::string& msg) = 0;

};

/**
 * Pointer to the logger
 */
PLASK_API extern shared_ptr<Logger> default_logger;

PLASK_API void createDefaultLogger();

/**
 * Log a message
 * \param level log level to log
 * \param msg log message
 **/
inline void writelog(LogLevel level, const std::string& msg) {
    if (!default_logger) createDefaultLogger();
    if (level <= maxLoglevel && (!default_logger->silent || level <= LOG_WARNING)) {
        default_logger->writelog(level, msg);
    }
}

/**
 * Log a message
 * \param level log level to log
 * \param msg log message
 * \param params parameters passed to format
 **/
template<typename... Args>
inline void writelog(LogLevel level, const std::string& msg, Args&&... params) {
    if (!default_logger) createDefaultLogger();
    if (level <= maxLoglevel && (!default_logger->silent || level <= LOG_WARNING)) {
        default_logger->writelog(level, format(msg, std::forward<Args>(params)...));
    }
}

}   // namespace plask

#ifdef NDEBUG
#   define write_debug(...) void(0)
#else
#   define write_debug(...) writelog(LOG_DEBUG, __VA_ARGS__)
#endif

#endif // PLASK__LOG_LOG_H
