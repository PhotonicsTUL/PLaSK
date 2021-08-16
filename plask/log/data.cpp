#include "data.hpp"

namespace plask {

template <> DataLog<std::string, std::string>&
DataLog<std::string, std::string>::operator()(const std::string& arg, const std::string& val, int counter) {
    writelog(LOG_DATA, "{}: {}: {}={} {}={} [{}]",
                global_prefix, chart_name, axis_arg_name, str(arg), axis_val_name, str(val), counter+1);
    return *this;
}

template <> DataLog<std::string, std::string>&
DataLog<std::string, std::string>::operator()(const std::string& arg, const std::string& val) {
    writelog(LOG_DATA, "{}: {}: {}={} {}={}",
                global_prefix, chart_name, axis_arg_name, str(arg), axis_val_name, str(val));
    return *this;
}

}
