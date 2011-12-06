#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectilinear.h>

BOOST_AUTO_TEST_SUITE(rectilinear_mesh)

BOOST_AUTO_TEST_CASE(dim1) {
    plask::RectilinearMesh1d mesh = {3.0, 1.0, 3.0};
    BOOST_CHECK_EQUAL(mesh.empty(), false);
    BOOST_REQUIRE_EQUAL(mesh.size(), 2);
    BOOST_CHECK_EQUAL(mesh[0], 1.0);
    BOOST_CHECK_EQUAL(mesh[1], 3.0);
    mesh.addPoints(1.0, 1.0, 3);
    BOOST_CHECK(mesh == plask::RectilinearMesh1d({1.0, 1.5, 2.0, 3.0}));   
    mesh.clear();
    BOOST_CHECK_EQUAL(mesh.empty(), true);
}

BOOST_AUTO_TEST_SUITE_END()