#ifndef PLASK__LOG_CHART_H
#define PLASK__LOG_CHART_H

#include <string>

namespace plask {

/**
 * Template of base class for classes which stores or log 2d chart data.
 */
template <typename ArgT, typename ValT>
struct Chart2d {

    /**
     * Append point to chart.
     * @param arg, val point to add
     * @return *this
     */
    virtual Chart2d& operator()(const ArgT& arg, const ValT& val) = 0;

};

/**
 * Send 2d chart data to logs.
 */
//TODO implementation
template <typename ArgT, typename ValT>
struct Chart2dLog: public Chart2d<ArgT, ValT> {

    Chart2dLog(const std::string& global_prefix, const std::string& chart_name, const std::string& axis_arg_name, const std::string& axis_val_name)
    {}

    Chart2dLog(const std::string& global_prefix, const std::string& axis_arg_name, const std::string& axis_val_name)
    {
        //chart_name = getUniqueString()
    }

    Chart2dLog& operator()(const ArgT& arg, const ValT& val)
    {}

};



}

#endif // PLASK__LOG_CHART_H
