#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectilinear.h>
#include <plask/mesh/boundary_conditions.h>

BOOST_AUTO_TEST_SUITE(boundary_conditions) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(boundary_conditions) {
    plask::BoundaryConditions<plask::RectilinearMesh2D, double> conditions;
    //conditions.add()
}

BOOST_AUTO_TEST_SUITE_END()
