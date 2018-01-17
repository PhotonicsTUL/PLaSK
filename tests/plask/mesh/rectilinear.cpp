#include <boost/test/unit_test.hpp>

#include <plask/geometry/stack.h>
#include <plask/geometry/leaf.h>

#include <plask/mesh/generator_rectangular.h>

BOOST_AUTO_TEST_SUITE(rectilinear) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(dim1) {
    plask::OrderedAxis mesh = {3.0, 1.0, 3.0};
    BOOST_CHECK_EQUAL(mesh.empty(), false);
    BOOST_REQUIRE_EQUAL(mesh.size(), 2);
    BOOST_CHECK_EQUAL(mesh[0], 1.0);
    BOOST_CHECK_EQUAL(mesh[1], 3.0);
    mesh.addPointsLinear(1.0, 2.0, 3);
    BOOST_CHECK_EQUAL(mesh, plask::OrderedAxis({1.0, 1.5, 2.0, 3.0}));
    double data[4] =  {0.7, 2.0, 3.0, 4.0};
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, 2.3), 3.3);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, 0.5), 0.7);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, 4.5), 4.0);
    mesh.clear();
    BOOST_CHECK_EQUAL(mesh.empty(), true);
}

BOOST_AUTO_TEST_CASE(dim2) {
    auto axis0 = plask::make_shared<plask::OrderedAxis>();
    auto axis1 = plask::make_shared<plask::OrderedAxis>();
    plask::RectangularMesh<2> mesh(axis0, axis1);
    BOOST_CHECK_EQUAL(mesh.empty(), true);
    axis0->addPointsLinear(0., 1., 2);
    axis1->addPointsLinear(0., 1., 2);
    BOOST_CHECK_EQUAL(mesh.size(), 4);
    double data[4] = {0.0, 2.0, 2.0, 0.0};
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(0.5, 0.5), plask::InterpolationFlags()), 1.0);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(-0.5, -0.5), plask::InterpolationFlags()), 0.0);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(1.5, -1.5), plask::InterpolationFlags()), 2.0);
    BOOST_CHECK_EQUAL(mesh.interpolateLinear(data, plask::vec(-1.5, 0.2), plask::InterpolationFlags()), 2.0 / 5.0);
    //mesh.clear();
    //BOOST_CHECK_EQUAL(mesh.empty(), true);
}

BOOST_AUTO_TEST_CASE(dim2boundary) {
    plask::RectangularMesh<2>::Boundary right_boundary = plask::RectangularMesh<2>::getRightBoundary();
    auto axis0 = plask::make_shared<plask::OrderedAxis>();
    auto axis1 = plask::make_shared<plask::OrderedAxis>();
    plask::RectangularMesh<2> mesh(axis0, axis1, plask::RectangularMesh<2>::ORDER_10);
    axis0->addPointsLinear(1.0, 3.0, 3);   //1.0, 2.0, 3.0
    axis1->addPointsLinear(5.0, 6.0, 2);   //5.0, 6.0
    auto right_boundary_on_mesh = right_boundary(mesh, plask::shared_ptr<plask::Geometry2DCartesian>());
    std::size_t expected[2] = { 2, 5 };
    BOOST_CHECK_EQUAL_COLLECTIONS(right_boundary_on_mesh.begin(), right_boundary_on_mesh.end(),
                                  std::begin(expected), std::end(expected));
    BOOST_CHECK(right_boundary_on_mesh.contains(2));
    BOOST_CHECK(!right_boundary_on_mesh.contains(1));
}

BOOST_AUTO_TEST_CASE(from_geometry_2) {
    // Prepare the geometry
    plask::shared_ptr<plask::StackContainer<2>> stack(new plask::StackContainer<2>);
    plask::shared_ptr<plask::Rectangle> rect1(new plask::Rectangle(plask::Vec<2>(2., 3.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Rectangle> rect2(new plask::Rectangle(plask::Vec<2>(4., 5.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Rectangle> rect3(new plask::Rectangle(plask::Vec<2>(2., 2.), plask::shared_ptr<plask::Material>()));
    stack->push_back(rect1, plask::align::center(0.0));
    stack->push_back(rect2, plask::align::center(0.0));
    stack->push_back(rect3, plask::align::center(0.0));

    auto mesh = plask::RectangularMesh2DSimpleGenerator().generate_t<plask::RectangularMesh<2>>(stack);
    BOOST_CHECK_EQUAL(*mesh->axis0, plask::OrderedAxis({-2., -1., 1., 2.}));
    BOOST_CHECK_EQUAL(*mesh->axis1, plask::OrderedAxis({0., 3., 8., 10.}));
}

BOOST_AUTO_TEST_CASE(from_geometry_3) {
    // Prepare the geometry
    plask::shared_ptr<plask::StackContainer<3>> stack(new plask::StackContainer<3>);
    plask::shared_ptr<plask::Cuboid> cub1(new plask::Cuboid(plask::Vec<3>(2., 2., 3.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Cuboid> cub2(new plask::Cuboid(plask::Vec<3>(4., 2., 5.), plask::shared_ptr<plask::Material>()));
    plask::shared_ptr<plask::Cuboid> cub3(new plask::Cuboid(plask::Vec<3>(2., 2., 2.), plask::shared_ptr<plask::Material>()));
    stack->push_back(cub1, plask::align::lonCenter(0.0) & plask::align::tranCenter(0.0));
    stack->push_back(cub2, plask::align::lonCenter(0.0) & plask::align::tranCenter(0.0));
    stack->push_back(cub3, plask::align::lonCenter(0.0) & plask::align::tranCenter(0.0));

    auto mesh = plask::RectangularMesh3DSimpleGenerator().generate_t<plask::RectangularMesh<3>>(stack);
    BOOST_CHECK_EQUAL(*mesh->axis0, plask::OrderedAxis({-2., -1., 1., 2.}));
    BOOST_CHECK_EQUAL(*mesh->axis1, plask::OrderedAxis({-1., 1.}));
    BOOST_CHECK_EQUAL(*mesh->axis2, plask::OrderedAxis({0., 3., 8., 10.}));
}

BOOST_AUTO_TEST_CASE(middle2) {
    auto axis0 = plask::make_shared<plask::OrderedAxis>();
    auto axis1 = plask::make_shared<plask::OrderedAxis>();
    plask::RectangularMesh<2> mesh(axis0, axis1);
    BOOST_CHECK_EQUAL(mesh.empty(), true);
    axis0->addPointsLinear(0., 4.0, 3);
    axis1->addPointsLinear(2., 6.0, 3);

    auto middles = mesh.getMidpointsMesh();
    BOOST_CHECK_EQUAL(*middles->axis0, plask::OrderedAxis({1., 3.}));
    BOOST_CHECK_EQUAL(*middles->axis1, plask::OrderedAxis({3., 5.}));
}

BOOST_AUTO_TEST_CASE(boundary) {
    auto axis0 = plask::make_shared<plask::OrderedAxis>();
    auto axis1 = plask::make_shared<plask::OrderedAxis>();
    plask::RectangularMesh<2> mesh(axis0, axis1, plask::RectangularMesh<2>::ORDER_10);
    axis0->addPointsLinear(0.0, 4.0, 3);
    axis1->addPointsLinear(2.0, 6.0, 3);
    auto leftB = plask::RectangularMesh<2>::getLeftBoundary();
    auto left = leftB(mesh, plask::shared_ptr<plask::Geometry2DCartesian>());
    // auto left = plask::RectilinearMesh2D::getLeftBoundary()(mesh); // WRONG!
    std::vector<std::size_t> indxs(left.begin(), left.end());

    BOOST_CHECK_EQUAL(indxs.size(), 3);
    BOOST_CHECK_EQUAL(indxs[0], 0);
    BOOST_CHECK_EQUAL(indxs[1], 3);
    BOOST_CHECK_EQUAL(indxs[2], 6);

    auto only_index_24 = plask::makePredicateBoundary<plask::RectangularMesh<2>>(
                [](const plask::RectangularMesh<2>& /*mesh*/, std::size_t index) -> bool { return index == 2 || index == 4; }
    );
    std::size_t expected[2] = { 2, 4 };
    auto pred_bound_with_mesh = only_index_24(mesh, plask::shared_ptr<plask::Geometry2DCartesian>());
    BOOST_CHECK_EQUAL_COLLECTIONS(pred_bound_with_mesh.begin(), pred_bound_with_mesh.end(),
                                  std::begin(expected), std::end(expected));

}

BOOST_AUTO_TEST_CASE(generator) {
    plask::RectangularMeshDivideGenerator<2> generator;

    auto stack(plask::make_shared<plask::StackContainer<2>>());
    stack->push_back(plask::make_shared<plask::Rectangle>(plask::Vec<2>(1., 4.), plask::shared_ptr<plask::Material>()));
    stack->push_back(plask::make_shared<plask::Rectangle>(plask::Vec<2>(1., 1.), plask::shared_ptr<plask::Material>()));
    stack->push_back(plask::make_shared<plask::Rectangle>(plask::Vec<2>(1., 8.), plask::shared_ptr<plask::Material>()));

    auto mesh = generator.get<plask::RectangularMesh<2>>(stack);
    BOOST_CHECK_EQUAL(mesh->axis1->at(0),  0.);
    BOOST_CHECK_EQUAL(mesh->axis1->at(1),  2.);
    BOOST_CHECK_EQUAL(mesh->axis1->at(2),  4.);
    BOOST_CHECK_EQUAL(mesh->axis1->at(3),  5.);
    BOOST_CHECK_EQUAL(mesh->axis1->at(4),  7.);
    BOOST_CHECK_EQUAL(mesh->axis1->at(5),  9.);
    BOOST_CHECK_EQUAL(mesh->axis1->at(6),  13.);
}

BOOST_AUTO_TEST_CASE(elements) {
    plask::RectangularMesh2DSimpleGenerator generator;
    auto stack(plask::make_shared<plask::StackContainer<2>>());
    stack->push_back(plask::make_shared<plask::Rectangle>(plask::Vec<2>(1., 4.), plask::shared_ptr<plask::Material>()));
    stack->push_back(plask::make_shared<plask::Rectangle>(plask::Vec<2>(1., 1.), plask::shared_ptr<plask::Material>()));
    stack->push_back(plask::make_shared<plask::Rectangle>(plask::Vec<2>(1., 8.), plask::shared_ptr<plask::Material>()));
    auto mesh = generator.get<plask::RectangularMesh<2>>(stack);

    size_t n = 0;
    for (auto elem = mesh->elements.begin(); elem != mesh->elements.end(); ++elem, ++n) {
        BOOST_CHECK_EQUAL(elem->getIndex(), n);
        BOOST_CHECK_EQUAL(elem->getLoLoIndex(), 2*n);
        BOOST_CHECK_EQUAL(elem->getUpLoIndex(), 2*n+1);
        BOOST_CHECK_EQUAL(elem->getLoUpIndex(), 2*n+2);
        BOOST_CHECK_EQUAL(elem->getUpUpIndex(), 2*n+3);
    }
}


BOOST_AUTO_TEST_SUITE_END()
