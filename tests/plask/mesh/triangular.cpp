#include <boost/test/unit_test.hpp>

#include <plask/mesh/triangular2d.h>
#include <plask/mesh/generator_triangular.h>

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
    /*
     0  1  2  3  4  5

     00-01-02-03-04-05  0
     | \| \| \| \| \|
     06-07-08-09-10-11  1
     | \| \| \| \| \|
     12-13-14 15-16-17  2
     | \| \     \| \|
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
    {
        plask::BoundaryNodeSet allBoundaries = mesh.getAllBoundary().get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(allBoundaries.size(), 5*4 /*outer*/ + 4 /*inner*/);
        BOOST_CHECK_EQUAL_COLLECTIONS(allBoundaries.begin(), allBoundaries.end(), all_boundary_indices.begin(), all_boundary_indices.end());
    }
    {
        plask::BoundaryNodeSet allBoundariesIn = mesh.getAllBoundaryIn(plask::Box2D(0.5, 0.5, 4.5, 4.5)).get(mesh, plask::make_shared<plask::Geometry2DCartesian>());
        BOOST_CHECK_EQUAL(allBoundariesIn.size(), 3*4 /*outer*/ + 4 /*inner*/);
        //BOOST_CHECK_EQUAL_COLLECTIONS(allBoundariesIn.begin(), allBoundariesIn.end(), all_boundary_indices.begin(), all_boundary_indices.end());
    }
    {
        //plask::BoundaryNodeSet leftBoundary = mesh.
    }
}
