/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
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
