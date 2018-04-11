#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectangular_filtered.h>

BOOST_AUTO_TEST_SUITE(rectangular_filtered) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(rectangular_filtered_2D) {
    // TODO
    plask::RectangularMesh<2> fullMesh;
    plask::RectangularFilteredMesh2D filterdMesh(
                &fullMesh,
                [] (const plask::RectangularMesh<2>::Element&) {
                    return true;
                }
    );
}

BOOST_AUTO_TEST_SUITE_END()
