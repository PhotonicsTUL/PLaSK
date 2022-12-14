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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Your solver test"
#include <boost/test/unit_test.hpp>

// This is needed because of a bug in Boost linkage in the newest Ubuntu
#if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
namespace boost { namespace unit_test { namespace ut_detail {
std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}
#endif

#include "../your_solver.hpp"
using namespace plask;

BOOST_AUTO_TEST_SUITE(your_solver)

BOOST_AUTO_TEST_CASE(your_test)
{
}


BOOST_AUTO_TEST_SUITE_END()
