#include <boost/test/unit_test.hpp>

#include <plask/mesh/triangular2d.h>
#include <plask/mesh/generator_triangular.h>

BOOST_AUTO_TEST_SUITE(triangular) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(triangular2d_simple) {
    plask::TriangularMesh2D mesh;
    BOOST_CHECK(mesh.empty());
    BOOST_CHECK(mesh.elements().empty());
    plask::TriangularMesh2D::Builder(mesh)
            .add(plask::vec(0.0, 0.0), plask::vec(1.0, 0.0), plask::vec(0.0, 1.0))
            .add(plask::vec(1.0, 0.0), plask::vec(0.0, 1.0), plask::vec(1.0, 1.0));
    BOOST_CHECK_EQUAL(mesh.size(), 4);
    BOOST_CHECK_EQUAL(mesh.elements().size(), 2);
}

BOOST_AUTO_TEST_CASE(triangular2d_boundaries) {
    auto node = [](std::size_t x_i, std::size_t y_i) { return x_i*6 + y_i; };
    plask::TriangularMesh2D mesh;
    std::vector<std::size_t> all_boundary_indices;
    /* Our mesh (has rectangular hole 14-15-21-20):

     0  1  2  3  4  5

     00-01-02-03-04-05  0
     | \| \| \| \| \|
     06-07-08-09-10-11  1
     | \| \| \| \| \|
     12-13-14-15-16-17  2
     | \| \|**| \| \|
     18-19-20-21-22-23  3
     | \| \| \| \| \|
     24-25-26-27-28-29  4
     | \| \| \| \| \|
     30-31-32-33-34-35  5
     */
    for (std::size_t x_i = 0; x_i < 6; ++x_i)
        for (std::size_t y_i = 0; y_i < 6; ++y_i) {
            mesh.nodes.emplace_back(x_i, y_i);
            if (x_i > 0 && y_i > 0 &&
              !(x_i == 3 && y_i == 3))  // rectangualar hole in a middle
            {
                mesh.elementNodes.push_back({node(x_i, y_i), node(x_i-1, y_i), node(x_i-1, y_i-1)});
                mesh.elementNodes.push_back({node(x_i, y_i), node(x_i, y_i-1), node(x_i-1, y_i-1)});
            }
            if (x_i == 0 || y_i == 0 || x_i == 5 || y_i == 5 ||
               (x_i==2&&y_i==2) || (x_i==2&&y_i==3) || (x_i==3&&y_i==2) || (x_i==3&&y_i==3))
                all_boundary_indices.push_back(node(x_i, y_i));
        }
    BOOST_CHECK_EQUAL(mesh.size(), 6*6);
    BOOST_CHECK_EQUAL(mesh.elements().size(), 5*5*2-2);
    // ---------------------- All boundaries ----------------------
    {
        plask::BoundaryNodeSet allBoundaries = mesh.getAllBoundary().get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(allBoundaries.size(), 5*4 /*outer*/ + 4 /*inner*/);
        BOOST_CHECK_EQUAL_COLLECTIONS(allBoundaries.begin(), allBoundaries.end(), all_boundary_indices.begin(), all_boundary_indices.end());
    }
    {
        plask::BoundaryNodeSet allBoundariesIn = mesh.getAllBoundaryIn(plask::Box2D(0.5, 0.5, 4.5, 4.5)).get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(allBoundariesIn.size(), 3*4 /*outer*/ + 4 /*inner*/);
        std::size_t expected[] = {7,8,9,10, 13,14,15,16, 19,20,21,22, 25,26,27,28};
        BOOST_CHECK_EQUAL_COLLECTIONS(allBoundariesIn.begin(), allBoundariesIn.end(), std::begin(expected), std::end(expected));
    }
    // --------------------- Left boundaries ----------------------
    {
        plask::BoundaryNodeSet leftBoundary = mesh.getLeftBoundary().get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(leftBoundary.size(), 6);
        std::size_t expected[] = {0, 6, 12, 18, 24, 30};
        BOOST_CHECK_EQUAL_COLLECTIONS(leftBoundary.begin(), leftBoundary.end(), std::begin(expected), std::end(expected));
    }
    {
        plask::BoundaryNodeSet leftBoundaryIn = mesh.getLeftOfBoundary(plask::Box2D(0.5, 0.5, 4.5, 4.5)).get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(leftBoundaryIn.size(), 4);
        std::size_t expected[] = {7, 13, 19, 25};
        BOOST_CHECK_EQUAL_COLLECTIONS(leftBoundaryIn.begin(), leftBoundaryIn.end(), std::begin(expected), std::end(expected));
    }
    // --------------------- Right boundaries ----------------------
    {
        plask::BoundaryNodeSet rightBoundary = mesh.getRightBoundary().get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(rightBoundary.size(), 6);
        std::size_t expected[] = {5, 11, 17, 23, 29, 35};
        BOOST_CHECK_EQUAL_COLLECTIONS(rightBoundary.begin(), rightBoundary.end(), std::begin(expected), std::end(expected));
    }
    {
        plask::BoundaryNodeSet rightBoundaryIn = mesh.getRightOfBoundary(plask::Box2D(0.5, 0.5, 4.5, 4.5)).get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(rightBoundaryIn.size(), 4);
        std::size_t expected[] = {10, 16, 22, 28};
        BOOST_CHECK_EQUAL_COLLECTIONS(rightBoundaryIn.begin(), rightBoundaryIn.end(), std::begin(expected), std::end(expected));
    }
    // --------------------- Bottom boundaries ---------------------
    {
        plask::BoundaryNodeSet bottomBoundary = mesh.getBottomBoundary().get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(bottomBoundary.size(), 6);
        std::size_t expected[] = {0, 1, 2, 3, 4, 5};
        BOOST_CHECK_EQUAL_COLLECTIONS(bottomBoundary.begin(), bottomBoundary.end(), std::begin(expected), std::end(expected));
    }
    {
        plask::BoundaryNodeSet bottomBoundaryIn = mesh.getBottomOfBoundary(plask::Box2D(0.5, 0.5, 4.5, 4.5)).get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(bottomBoundaryIn.size(), 4);
        std::size_t expected[] = {7, 8, 9, 10};
        BOOST_CHECK_EQUAL_COLLECTIONS(bottomBoundaryIn.begin(), bottomBoundaryIn.end(), std::begin(expected), std::end(expected));
    }
    // --------------------- Top boundaries ------------------------
    {
        plask::BoundaryNodeSet topBoundary = mesh.getTopBoundary().get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(topBoundary.size(), 6);
        std::size_t expected[] = {30, 31, 32, 33, 34, 35};
        BOOST_CHECK_EQUAL_COLLECTIONS(topBoundary.begin(), topBoundary.end(), std::begin(expected), std::end(expected));
    }
    {
        plask::BoundaryNodeSet topBoundaryIn = mesh.getTopOfBoundary(plask::Box2D(0.5, 0.5, 4.5, 4.5)).get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(topBoundaryIn.size(), 4);
        std::size_t expected[] = {25, 26, 27, 28};
        BOOST_CHECK_EQUAL_COLLECTIONS(topBoundaryIn.begin(), topBoundaryIn.end(), std::begin(expected), std::end(expected));
    }
}

BOOST_AUTO_TEST_SUITE_END()
