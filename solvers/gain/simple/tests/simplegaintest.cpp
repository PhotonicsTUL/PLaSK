#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Simple gain test"
#include <boost/test/unit_test.hpp>

#include "../fermi.h"
using namespace plask;
using namespace plask::solvers::fermi;

#ifdef _WIN32
#   define V "\\"
#else
#   define V "/"
#endif

BOOST_AUTO_TEST_SUITE(gain)

BOOST_AUTO_TEST_CASE(detect_active_region)
{
    MaterialsDB::loadAllToDefault(prefixPath() + V ".." V ".." V ".." V "lib" V "plask" V "materials");
    MaterialsDB& materials = MaterialsDB::getDefault();

    auto stack = make_shared<StackContainer<2>>();

    auto barrier = make_shared<Block<2>>(Vec<2>(10., 5.), materials.get("GaAs"));
    auto well = make_shared<Block<2>>(Vec<2>(10., 7.), materials.get("In(0.3)GaAs"));
    well->addRole("QW");

    auto active1 = make_shared<MultiStackContainer<2>>(2);
    active1->addRole("active");
    active1->push_front(barrier);
    active1->push_front(well);
    active1->push_front(barrier);

    auto cladding = make_shared<Block<2>>(Vec<2>(10., 20.), materials.get("Al(0.1)GaAs"));

    auto nothing = make_shared<Block<2>>(Vec<2>(2., 5.));
    auto shelf = make_shared<ShelfContainer2D>();
    shelf->push_back(nothing);
    shelf->push_back(active1);

    stack->push_front(cladding);
    stack->push_front(active1);
    stack->push_front(cladding);
    stack->push_front(shelf);
    stack->push_front(cladding);

    auto geometry = make_shared<Geometry2DCartesian>(stack, 1000.);

    FermiGainSolver<Geometry2DCartesian> solver("gaintest");

    solver.setGeometry(geometry);
    solver.compute();

    BOOST_CHECK_EQUAL(solver.regions.size(), 2);
    BOOST_CHECK_EQUAL(solver.regions[0].origin, Vec<2>(2., 20.));
    BOOST_CHECK_EQUAL(solver.regions[1].origin, Vec<2>(0., 74.));

    BOOST_CHECK_EQUAL(solver.regions[0].layers->getChildrenCount(), 5);
    BOOST_CHECK_EQUAL(solver.regions[0].isQW.size(), 5);

    std::vector<bool> qw = { false, true, false, true, false };
    BOOST_CHECK(solver.regions[0].isQW == qw);

    BOOST_CHECK_EQUAL(solver.regions[0].getLayerMaterial(1), well->material);
    BOOST_CHECK_EQUAL(solver.regions[0].getLayerBox(1), Box2D(2., 25., 12., 32.));
    BOOST_CHECK_EQUAL(solver.regions[0].getBoundingBox(), Box2D(2., 20., 12., 54.));
    BOOST_CHECK_EQUAL(solver.regions[1].getLayerBox(1), Box2D(0., 79., 10., 86.));
}

BOOST_AUTO_TEST_SUITE_END()
