#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectilinear.h>
#include <plask/geometry/stack.h>
#include <plask/geometry/leaf.h>

BOOST_AUTO_TEST_SUITE(rectilinear) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(dim1) {
    plask::RectilinearMesh1d mesh = {3.0, 1.0, 3.0};
    BOOST_CHECK_EQUAL(mesh.empty(), false);
    BOOST_REQUIRE_EQUAL(mesh.size(), 2);
    BOOST_CHECK_EQUAL(mesh[0], 1.0);
    BOOST_CHECK_EQUAL(mesh[1], 3.0);
    mesh.addPointsLinear(1.0, 2.0, 3);
    BOOST_CHECK_EQUAL(mesh, plask::RectilinearMesh1d({1.0, 1.5, 2.0, 3.0}));
    double data[4] =  {0.7, 2.0, 3.0, 4.0};
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, 2.3), 3.3);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, 0.5), 0.7);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, 4.5), 4.0);
    mesh.clear();
    BOOST_CHECK_EQUAL(mesh.empty(), true);
}

BOOST_AUTO_TEST_CASE(dim2) {
    plask::RectilinearMesh2d mesh;
    BOOST_CHECK_EQUAL(mesh.empty(), true);
    mesh.c0.addPointsLinear(0, 1.0, 2);
    mesh.c1.addPointsLinear(0, 1.0, 2);
    BOOST_CHECK_EQUAL(mesh.size(), 4);
    double data[4] = {0.0, 2.0, 2.0, 0.0};
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(0.5, 0.5)), 1.0);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(-0.5, -0.5)), 0.0);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(1.5, -1.5)), 2.0);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(-1.5, 0.2)), 2.0 / 5.0);
    mesh.clear();
    BOOST_CHECK_EQUAL(mesh.empty(), true);
}

BOOST_AUTO_TEST_CASE(from_geometry_2) {
    // Prepare the geometry
    plask::shared_ptr<plask::StackContainer<2>> stack(new plask::StackContainer<2>);
    plask::shared_ptr<plask::Rectangle> rect1(new plask::Rectangle(plask::Vec<2>(2., 3.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Rectangle> rect2(new plask::Rectangle(plask::Vec<2>(4., 5.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Rectangle> rect3(new plask::Rectangle(plask::Vec<2>(2., 2.), plask::shared_ptr<plask::Material>()));
    stack->push_back(rect1);
    stack->push_back(rect2);
    stack->push_back(rect3);

    plask::RectilinearMesh2d mesh(stack);
    BOOST_CHECK_EQUAL(mesh.c0, plask::RectilinearMesh1d({-2., -1., 1., 2.}));
    BOOST_CHECK_EQUAL(mesh.c1, plask::RectilinearMesh1d({0., 3., 8., 10.}));

}

BOOST_AUTO_TEST_CASE(from_geometry_3) {
    // Prepare the geometry
    plask::shared_ptr<plask::StackContainer<3>> stack(new plask::StackContainer<3>);
    plask::shared_ptr<plask::Cuboid> cub1(new plask::Cuboid(plask::Vec<3>(2., 2., 3.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Cuboid> cub2(new plask::Cuboid(plask::Vec<3>(4., 2., 5.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Cuboid> cub3(new plask::Cuboid(plask::Vec<3>(2., 2., 2.), plask::shared_ptr<plask::Material>()));
    stack->push_back(cub1);
    stack->push_back(cub2);
    stack->push_back(cub3);

    plask::RectilinearMesh3d mesh(stack);
    BOOST_CHECK_EQUAL(mesh.c0, plask::RectilinearMesh1d({-2., -1., 1., 2.}));
    BOOST_CHECK_EQUAL(mesh.c1, plask::RectilinearMesh1d({-1., 1.}));
    BOOST_CHECK_EQUAL(mesh.c2, plask::RectilinearMesh1d({0., 3., 8., 10.}));

}

BOOST_AUTO_TEST_SUITE_END()