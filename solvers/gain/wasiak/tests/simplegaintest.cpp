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
#define BOOST_TEST_MODULE "Simple gain test"
#include <boost/test/unit_test.hpp>

#include "../fermi.hpp"
using namespace plask;
using namespace plask::solvers::fermi;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define V "\\"
#else
#   define V "/"
#endif

#if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
namespace boost { namespace unit_test { namespace ut_detail {
std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}
#endif

struct TheSolver: public FermiGainSolver<Geometry2DCartesian>
{
    TheSolver(const std::string name=""): FermiGainSolver<Geometry2DCartesian>(name) {}

   void detect_active_regions() { detectActiveRegions(); }
};

BOOST_AUTO_TEST_SUITE(gain)

BOOST_AUTO_TEST_CASE(detect_active_region)
{
    MaterialsDB::loadAllToDefault(prefixPath() + V ".." V ".." V ".." V "lib" V "plask" V "materials");
    MaterialsDB& materials = MaterialsDB::getDefault();

    auto stack = plask::make_shared<StackContainer<2>>();

    auto barrier = plask::make_shared<Block<2>>(Vec<2>(10., 5.), materials.get("GaAs"));
    auto well = plask::make_shared<Block<2>>(Vec<2>(10., 7.), materials.get("In(0.3)GaAs"));
    well->addRole("QW");

    auto active1 = plask::make_shared<MultiStackContainer<plask::StackContainer<2>>>(2);
    active1->addRole("active");
    active1->push_front(barrier);
    active1->push_front(well);
    active1->push_front(barrier);

    auto cladding1 = plask::make_shared<Block<2>>(Vec<2>(10., 20.), materials.get("Al(0.1)GaAs"));
    auto cladding2 = plask::make_shared<Block<2>>(Vec<2>(10., 20.), materials.get("Al(0.1)GaAs"));

    auto nothing = plask::make_shared<Block<2>>(Vec<2>(2., 5.));
    auto shelf1 = plask::make_shared<ShelfContainer2D>();
    shelf1->push_back(nothing);
    shelf1->push_back(active1);

    auto active2 = plask::make_shared<StackContainer<2>>();
    active2->addRole("active");
    active2->push_back(well);

    stack->push_front(cladding2);
    stack->push_front(active2);
    stack->push_front(cladding1);
    stack->push_front(active1);
    stack->push_front(cladding2);
    stack->push_front(shelf1);
    stack->push_front(cladding1);

    auto geometry = plask::make_shared<Geometry2DCartesian>(stack, 1000.);

    TheSolver solver("gaintest");

    solver.setGeometry(geometry);
    solver.detect_active_regions();

    BOOST_CHECK_EQUAL(solver.regions.size(), 3);
    BOOST_CHECK_EQUAL(solver.regions[0].origin, Vec<2>(2., 20.));
    BOOST_CHECK_EQUAL(solver.regions[1].origin, Vec<2>(0., 74.));

    BOOST_CHECK_EQUAL(solver.regions[0].layers->getChildrenCount(), 5);
    BOOST_CHECK_EQUAL(solver.regions[0].size(), 5);

    std::vector<bool> qw = { false, true, false, true, false };
    for (size_t i = 0; i < 5; ++i)
        BOOST_CHECK_EQUAL(solver.regions[0].isQW(i), qw[i]);

    BOOST_CHECK_EQUAL(solver.regions[0].getLayerMaterial(1), well->singleMaterial());
    BOOST_CHECK_EQUAL(solver.regions[0].getLayerBox(1), Box2D(2., 25., 12., 32.));
    BOOST_CHECK_EQUAL(solver.regions[0].getBoundingBox(), Box2D(2., 20., 12., 54.));
    BOOST_CHECK_EQUAL(solver.regions[1].getLayerBox(1), Box2D(0., 79., 10., 86.));

    BOOST_CHECK_EQUAL(solver.regions[2].size(), 3);
    BOOST_CHECK_EQUAL(solver.regions[2].getLayerMaterial(0), cladding1->singleMaterial());
    BOOST_CHECK_EQUAL(solver.regions[2].getLayerMaterial(2), cladding2->singleMaterial());
}

BOOST_AUTO_TEST_SUITE_END()
