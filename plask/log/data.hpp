#ifndef PLASK__LOG_DATA_H
#define PLASK__LOG_DATA_H

#include <string>

#include "log.hpp"
#include "id.hpp"

namespace plask {

/**
 * Template of base class for classes which store or log n-dimensional data
 */
template <typename ArgT, typename ValT>
class DataLog {

    int cntr;                   ///< Counter for counted uses
    std::string global_prefix;  ///< Prefix for the log
    std::string chart_name;     ///< Name of the plot
    std::string axis_arg_name;  ///< Name of the argument
    std::string axis_val_name;  ///< Name of the value

  protected:

    /**
     * Log a data point and with specified counter. Most probably add another point to the list.
     * @param counter current counter value
     * @param data data to log (e.g. argument and value)
     * @return current counter
     */
    DataLog& operator()(const ArgT& arg, const ValT& val, int counter) {
        writelog(LOG_DATA, "{}: {}: {}={} {}={} ({}) [{}]",
                 global_prefix, chart_name, axis_arg_name, str(arg), axis_val_name, str(val), str(abs(val)), counter+1);
        return *this;
    };

  public:

    DataLog(const std::string& global_prefix, const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name) :
        cntr(0), global_prefix(global_prefix), chart_name(chart_name), axis_arg_name(axis_arg_name), axis_val_name(axis_val_name)
    {
    }

    DataLog(const std::string& global_prefix, const std::string& axis_arg_name, const std::string& axis_val_name) :
        cntr(0), global_prefix(global_prefix), chart_name(axis_val_name + "(" + axis_arg_name + ")_" + getUniqueString()), axis_arg_name(axis_arg_name), axis_val_name(axis_val_name)
    {
        // chart_name = axis_arg_name(axis_arg_name)_getUniqueString()
    }

    /**
     * Log a data point. Most probably add another point to the chart.
     * @param data data to log (e.g. argument and value)
     * @return *this
     */
    DataLog& operator()(const ArgT& arg, const ValT& val) {
        writelog(LOG_DATA, "{}: {}: {}={} {}={} ({})",
                 global_prefix, chart_name, axis_arg_name, str(arg), axis_val_name, str(val), str(abs(val)));
        return *this;
    }

    /**
     * Log a data point with automatic counting. Most probably add another point to the list.
     * @param data data to log (e.g. argument and value)
     * @return current counter
     */
    inline int count(const ArgT& arg, const ValT& val) { (*this)(arg, val, cntr); return ++cntr; };


    /// Return current counter
    int counter() const { return cntr; }

    /// Reset the counter
    void resetCounter() { cntr = 0; }

    /// Report and throw error
    void throwError(const ArgT& arg) const {
        writelog(LOG_ERROR_DETAIL, "{0}: {4}: {1}={3} {2}=ERROR", global_prefix, axis_arg_name, axis_val_name, str(arg), chart_name);
        throw;
    }

    /// Return chart name
    std::string chartName() const { return chart_name; }
};

template <> DataLog<std::string, std::string>&
DataLog<std::string, std::string>::operator()(const std::string& arg, const std::string& val, int counter);

template <> DataLog<std::string, std::string>&
DataLog<std::string, std::string>::operator()(const std::string& arg, const std::string& val);

}

#endif // PLASK__LOG_DATA_H
