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
#ifndef PLASK__VECTOR__BOOST_GEOMETRY_H
#define PLASK__VECTOR__BOOST_GEOMETRY_H

/** @file
This file contains adaptation of plask's vector for using it with boost geometry.
*/

#include "2d.hpp"

#include <boost/geometry.hpp>

// code is based on:
// https://www.boost.org/doc/libs/1_61_0/libs/geometry/doc/html/geometry/examples/example_source_code__adapting_a_legacy_geometry_object_model.html#adaption_of_qpoint_source_code
namespace boost
{
    namespace geometry
    {
        namespace traits
        {
            // Adapt plask::Vec<2, double> to Boost.Geometry

            template<> struct tag<plask::Vec<2, double>>
            { typedef point_tag type; };

            template<> struct coordinate_type<plask::Vec<2, double>>
            { typedef double type; };

            template<> struct coordinate_system<plask::Vec<2, double>>
            { typedef cs::cartesian type; };

            template<> struct dimension<plask::Vec<2, double>> : boost::mpl::int_<2> {};

            template<>
            struct access<plask::Vec<2, double>, 0>
            {
                static double get(plask::Vec<2, double> const& p)
                {
                    return p.c0;
                }

                static void set(plask::Vec<2, double>& p, double const& value)
                {
                    p.c0 = value;
                }
            };

            template<>
            struct access<plask::Vec<2, double>, 1>
            {
                static double get(plask::Vec<2, double> const& p)
                {
                    return p.c1;
                }

                static void set(plask::Vec<2, double>& p, double const& value)
                {
                    p.c1 = value;
                }
            };
        }
    }
} // namespace boost::geometry::traits

#endif // PLASK__VECTOR__BOOST_GEOMETRY_H
