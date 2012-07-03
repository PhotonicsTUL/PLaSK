#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectilinear.h>
#include <plask/mesh/boundary_conditions.h>

BOOST_AUTO_TEST_SUITE(boundary_conditions) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(boundary_conditions) {
    plask::BoundaryConditions<plask::RectilinearMesh2D, double> conditions;
    BOOST_CHECK(conditions.empty());
    conditions.add(plask::RectilinearMesh2D::getLeftBoundary(), 1.0);
    conditions.add(plask::RectilinearMesh2D::getRightBoundary(), 2.0);
    BOOST_CHECK_EQUAL(conditions.size(), 2);
    BOOST_CHECK_EQUAL(conditions[0].condition, 1.0);

    plask::RectilinearMesh2D mesh;
    mesh.c0.addPointsLinear(1.0, 3.0, 3);   //1.0, 2.0, 3.0
    mesh.c1.addPointsLinear(5.0, 6.0, 2);   //5.0, 6.0
    BOOST_CHECK(conditions.includes(mesh, 0) == conditions.begin());

}

BOOST_AUTO_TEST_SUITE_END()
