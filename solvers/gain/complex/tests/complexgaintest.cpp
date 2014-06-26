#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Complex gain test"
#include <boost/test/unit_test.hpp>

#include "../ferminew.h"
using namespace plask;
using namespace plask::solvers::ferminew;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define V "\\"
#else
#   define V "/"
#endif

struct TheSolver: public FerminewGainSolver<Geometry2DCartesian>
{
    TheSolver(const std::string name=""): FerminewGainSolver<Geometry2DCartesian>(name) {}

   void detect_active_regions() { detectActiveRegions(); }
};

BOOST_AUTO_TEST_SUITE(gain)

BOOST_AUTO_TEST_CASE(detect_active_region)
{
    // nowy kod uwzgledniajacy to, ze nowe wzmocnienie dodaje do "region" warstwy sasiadujace z obszarem czynnym

    MaterialsDB::loadAllToDefault(prefixPath() + V ".." V ".." V ".." V "lib" V "plask" V "materials");
    MaterialsDB& materials = MaterialsDB::getDefault();

    auto stack = make_shared<StackContainer<2>>();

    auto substrate = make_shared<Block<2>>(Vec<2>(100., 500.), materials.get("GaAs"));
    auto cladding1 = make_shared<Block<2>>(Vec<2>(20., 20e-3), materials.get("Al(0.1)GaAs"));
    auto cladding2 = make_shared<Block<2>>(Vec<2>(20., 20e-3), materials.get("Al(0.1)GaAs"));
    auto cladding3 = make_shared<Block<2>>(Vec<2>(20., 20e-3), materials.get("Al(0.1)GaAs"));
    auto barrier = make_shared<Block<2>>(Vec<2>(20., 5e-3), materials.get("GaAs"));
    auto well = make_shared<Block<2>>(Vec<2>(20., 8e-3), materials.get("In(0.3)GaAs"));
    well->addRole("QW");
    substrate->addRole("substrate");

    auto active1 = make_shared<MultiStackContainer<2>>(); // info: (2) -> *2
    active1->addRole("active");
    active1->push_front(barrier);
    active1->push_front(well);
    active1->push_front(barrier);

    auto active2 = make_shared<MultiStackContainer<2>>();
    active2->addRole("active");
    active2->push_front(barrier);
    active2->push_front(well);
    active2->push_front(barrier);
    active2->push_front(well);
    active2->push_front(barrier);

    stack->push_front(cladding1);
    stack->push_front(active2);
    stack->push_front(cladding2);
    stack->push_front(active1);
    stack->push_front(cladding3);
    stack->push_front(substrate);

    auto geometry = make_shared<Geometry2DCartesian>(stack, 1000.);

    TheSolver solver("gaintest");

    solver.setGeometry(geometry);
    solver.detect_active_regions();

    BOOST_CHECK_EQUAL(solver.regions.size(), 2);
    BOOST_CHECK_EQUAL(solver.regions[0].origin, Vec<2>(0., 500.000)); // info: 500.020-0.020, -0.020 -> - cladding
    //BOOST_CHECK_EQUAL(solver.regions[1].origin, Vec<2>(0., 500.038)); // info: 500.058-0.020, -0.020 -> - cladding

    BOOST_CHECK_EQUAL(solver.regions[0].layers->getChildrenCount(), 3+2); // info: +2 -> + claddings
    BOOST_CHECK_EQUAL(solver.regions[0].size(), 3+2); // info: +2 -> + claddings
    BOOST_CHECK_EQUAL(solver.regions[1].layers->getChildrenCount(), 5+2); // info: +2 -> + claddings
    BOOST_CHECK_EQUAL(solver.regions[1].size(), 5+2); // info: +2 -> + claddings

    std::vector<bool> qw = { false, true, false };
    for (size_t i = 0; i < 3; ++i)
        BOOST_CHECK_EQUAL(solver.regions[1].isQW(i+1), qw[i]); // info: +1 -> + cladding

    qw.push_back(true);
    qw.push_back(false);
    for (size_t i = 0; i < 5; ++i)
        BOOST_CHECK_EQUAL(solver.regions[1].isQW(i+1), qw[i]); // info: +1 -> + cladding

    // stary kod

    /*MaterialsDB::loadAllToDefault(prefixPath() + V ".." V ".." V ".." V "lib" V "plask" V "materials");
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

    auto cladding1 = make_shared<Block<2>>(Vec<2>(10., 20.), materials.get("Al(0.1)GaAs"));
    auto cladding2 = make_shared<Block<2>>(Vec<2>(10., 20.), materials.get("Al(0.1)GaAs"));

    auto nothing = make_shared<Block<2>>(Vec<2>(2., 5.));
    auto shelf1 = make_shared<ShelfContainer2D>();
    shelf1->push_back(nothing);
    shelf1->push_back(active1);

    auto active2 = make_shared<StackContainer<2>>();
    active2->addRole("active");
    active2->push_back(well);

    stack->push_front(cladding2);
    stack->push_front(active2);
    stack->push_front(cladding1);
    stack->push_front(active1);
    stack->push_front(cladding2);
    stack->push_front(shelf1);
    stack->push_front(cladding1);

    auto geometry = make_shared<Geometry2DCartesian>(stack, 1000.);

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
    BOOST_CHECK_EQUAL(solver.regions[2].getLayerMaterial(2), cladding2->singleMaterial());*/
}

BOOST_AUTO_TEST_SUITE_END()
