#ifndef PLASK__LOG_DATA_H
#define PLASK__LOG_DATA_H

#include <string>

namespace plask {

/**
 * Template of base class for classes which store or log log n-dimensional data
 */
template <typename... Params>
struct DataLog {

    /**
     * Log a data point. Most probably add another point to the chart.
     * @param data data to log (e.g. argument and value)
     * @return *this
     */
    virtual DataLog& operator()(const Params&... data) = 0;

};

/**
 * Send 2d data to logs.
 */
//TODO implementation
template <typename ArgT, typename ValT>
struct Data2dLog: public DataLog<ArgT, ValT> {

    Data2dLog(const std::string& global_prefix, const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name) {
    }

    Data2dLog(const std::string& global_prefix, const std::string& axis_arg_name, const std::string& axis_val_name) {
        //chart_name = axis_arg_name(axis_arg_name)_getUniqueString()
    }

    virtual Data2dLog& operator()(const ArgT& arg, const ValT& val) {}

};



}

#endif // PLASK__LOG_DATA_H
