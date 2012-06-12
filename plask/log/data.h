#ifndef PLASK__LOG_DATA_H
#define PLASK__LOG_DATA_H

#include <string>

#include "log.h"
#include "id.h"

namespace plask {

/**
 * Template of base class for classes which store or log log n-dimensional data
 */
template <typename... Params>
class DataLog {

    /// Counter for counted uses
    int cntr;

  protected:

    /**
     * Log a data point and with specified counter. Most probably add another point to the list.
     * @param counter current counter value
     * @param data data to log (e.g. argument and value)
     * @return current counter
     */
      virtual DataLog& operator()(const Params&... data, int counter) = 0;


  public:

    DataLog() : cntr(0) {}

    /**
     * Log a data point. Most probably add another point to the chart.
     * @param data data to log (e.g. argument and value)
     * @return *this
     */
    virtual DataLog& operator()(const Params&... data) = 0;

    /**
     * Log a data point with automatic counting. Most probably add another point to the list.
     * @param data data to log (e.g. argument and value)
     * @return current counter
     */
    int count(const Params&... data) { (*this)(std::forward<const Params&>(data)..., cntr); return ++cntr; };

    /// Reset the counter
    void resetCounter() { cntr = 0; }




};

/**
 * Send 2d data to logs.
 */
//TODO implementation
template <typename ArgT, typename ValT>
struct Data2dLog: public DataLog<ArgT, ValT> {

    std::string global_prefix;  ///< Prefix for the log
    std::string chart_name;     ///< Name of the plot
    std::string axis_arg_name;  ///< Name of the argument
    std::string axis_val_name;  ///< Name of the value

    Data2dLog(const std::string& global_prefix, const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name) :
        global_prefix(global_prefix), chart_name(chart_name), axis_arg_name(axis_arg_name), axis_val_name(axis_val_name)
    {
    }

    Data2dLog(const std::string& global_prefix, const std::string& axis_arg_name, const std::string& axis_val_name) :
        global_prefix(global_prefix), chart_name(axis_arg_name + "_" + getUniqueString()), axis_arg_name(axis_arg_name), axis_val_name(axis_val_name)
    {
        //chart_name = axis_arg_name(axis_arg_name)_getUniqueString()
    }

    virtual Data2dLog& operator()(const ArgT& arg, const ValT& val) {
        writelog(LOG_DATA, "%1%: %2%=%4% %3%=%5%", chart_name, axis_arg_name, axis_val_name, str(arg), str(val));
        return *this;
    }

    virtual Data2dLog& operator()(const ArgT& arg, const ValT& val, int counter) {
        writelog(LOG_DATA, "%1%: %2%=%4% %3%=%5% (%6%)", chart_name, axis_arg_name, axis_val_name, str(arg), str(val), counter);
        return *this;
    };
};



}

#endif // PLASK__LOG_DATA_H
