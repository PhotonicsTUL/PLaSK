#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectangular_filtered.h>

void checkNodeIterator(const plask::RectangularFilteredMesh2D& filteredMesh,
                       plask::RectangularFilteredMesh2D::const_iterator& it,
                       std::size_t index, std::size_t number,
                       double x, double y)
{
    BOOST_REQUIRE(it != filteredMesh.end());
    BOOST_CHECK_EQUAL(it.getIndex(), index);
    BOOST_CHECK_EQUAL(it.getNumber(), number);
    BOOST_CHECK_EQUAL(*it, plask::vec(x, y));
    ++it;
}

// TODO more parameters to test
void checkElementIterator(const plask::RectangularFilteredMesh2D& filteredMesh,
                          plask::RectangularFilteredMesh2D::Elements::const_iterator& it,
                          std::size_t index, std::size_t number)
{
    BOOST_REQUIRE(it != filteredMesh.elements().end());
    BOOST_CHECK_EQUAL(it.getIndex(), index);
    BOOST_CHECK_EQUAL(it.getNumber(), number);
    ++it;
}

BOOST_AUTO_TEST_SUITE(rectangular_filtered) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(rectangular_filtered_2D) {
    auto axis0 = plask::make_shared<plask::OrderedAxis>(std::initializer_list<double>{1.0, 2.0, 5.0, 10.0});
    auto axis1 = plask::make_shared<plask::RegularAxis>(3.0, 6.0, 4);
    plask::RectangularMesh<2> fullMesh(axis0, axis1);   // 4x4 nodes, 3x3 elements
    plask::RectangularFilteredMesh2D filteredMesh(
                &fullMesh,
                [] (const plask::RectangularMesh<2>::Element& e) {
                    return e.getIndex0() == 1 || e.getIndex1() == 1;    // everything in the middle (index 1) column and row
                }
    );
    BOOST_REQUIRE_EQUAL(filteredMesh.size(), 2 + 4 + 4 + 2);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount(), 1 + 3 + 1);

    {   // iterator test:
        plask::RectangularFilteredMesh2D::const_iterator it = filteredMesh.begin();

        checkNodeIterator(filteredMesh, it,    0,  1,   1.0, 4.0);
        checkNodeIterator(filteredMesh, it,    1,  2,   1.0, 5.0);

        checkNodeIterator(filteredMesh, it,    2,  4,   2.0, 3.0);
        checkNodeIterator(filteredMesh, it,    3,  5,   2.0, 4.0);
        checkNodeIterator(filteredMesh, it,    4,  6,   2.0, 5.0);
        checkNodeIterator(filteredMesh, it,    5,  7,   2.0, 6.0);

        checkNodeIterator(filteredMesh, it,    6,  8,   5.0, 3.0);
        checkNodeIterator(filteredMesh, it,    7,  9,   5.0, 4.0);
        checkNodeIterator(filteredMesh, it,    8, 10,   5.0, 5.0);
        checkNodeIterator(filteredMesh, it,    9, 11,   5.0, 6.0);

        checkNodeIterator(filteredMesh, it,   10, 13,  10.0, 4.0);
        checkNodeIterator(filteredMesh, it,   11, 14,  10.0, 5.0);

        BOOST_CHECK(it == filteredMesh.end());
    }

    {   // element iterator test:
        plask::RectangularFilteredMesh2D::Elements::const_iterator it = filteredMesh.elements().begin();

        checkElementIterator(filteredMesh, it,   0,  1);

        checkElementIterator(filteredMesh, it,   1,  3);
        checkElementIterator(filteredMesh, it,   2,  4);
        checkElementIterator(filteredMesh, it,   3,  5);

        checkElementIterator(filteredMesh, it,   4,  7);

        BOOST_CHECK(it == filteredMesh.elements().end());
    }

}

BOOST_AUTO_TEST_SUITE_END()
