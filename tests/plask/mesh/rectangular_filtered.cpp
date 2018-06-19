#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectangular_filtered.h>

plask::RectangularFilteredMesh2D constructMesh(plask::RectangularMesh2D::IterationOrder iter_order = plask::RectangularMesh2D::ORDER_01) {
    auto axis0 = plask::make_shared<plask::OrderedAxis>(std::initializer_list<double>{1.0, 2.0, 5.0, 10.0, 18.0});
    auto axis1 = plask::make_shared<plask::RegularAxis>(3.0, 6.0, 4);
    return plask::RectangularFilteredMesh2D(
                plask::RectangularMesh2D(axis0, axis1, iter_order),    // 5x4 nodes, 4x3 elements
                [] (const plask::RectangularMesh<2>::Element& e) {
                    return e.getIndex0() == 1 || e.getIndex1() == 1 ||
                          (e.getIndex0() == 3 && e.getIndex1() == 2);
                }
    );
}

template <int DIM>
void checkNodeIterator(const plask::RectangularFilteredMesh<DIM>& filteredMesh,
                       typename plask::RectangularFilteredMesh<DIM>::const_iterator& it,
                       std::size_t index, std::size_t number,
                       plask::Vec<DIM, double> coords)
{
    BOOST_REQUIRE(it != filteredMesh.end());
    BOOST_CHECK_EQUAL(it.getIndex(), index);
    BOOST_CHECK_EQUAL(it.getNumber(), number);
    BOOST_CHECK_EQUAL(*it, coords);
    ++it;
}

// TODO more parameters to test
void checkElementIterator(const plask::RectangularFilteredMesh2D& filteredMesh,
                          plask::RectangularFilteredMesh2D::Elements::const_iterator& it,
                          std::size_t index, std::size_t number,
                          std::size_t index0, std::size_t index1)
{
    BOOST_REQUIRE(it != filteredMesh.elements().end());
    BOOST_CHECK_EQUAL(it.getIndex(), index);
    BOOST_CHECK_EQUAL(it->getIndex(), index);
    BOOST_CHECK_EQUAL(it.getNumber(), number);

    BOOST_CHECK_EQUAL(it->getIndex0(), index0);
    BOOST_CHECK_EQUAL(it->getLowerIndex0(), index0);
    BOOST_CHECK_EQUAL(it->getUpperIndex0(), index0+1);
    BOOST_CHECK_EQUAL(it->getIndex1(), index1);
    BOOST_CHECK_EQUAL(it->getLowerIndex1(), index1);
    BOOST_CHECK_EQUAL(it->getUpperIndex1(), index1+1);

    ++it;
}

void checkBoundary(const plask::BoundaryNodeSet& b, std::vector<std::size_t> expected) {
    BOOST_CHECK_EQUAL(b.size(), expected.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(b.begin(), b.end(), expected.begin(), expected.end());
    for (auto el: expected)
        BOOST_CHECK(b.contains(el));
    BOOST_CHECK(!b.contains(expected.empty() ? 0 : expected.back()+1));
}

BOOST_AUTO_TEST_SUITE(rectangular_filtered) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(rectangular_filtered_2D) {
    plask::RectangularFilteredMesh2D filteredMesh = constructMesh();
    BOOST_REQUIRE_EQUAL(filteredMesh.size(), 2 + 5 + 5 + 4);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount(), 1 + 4 + 2);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount0(), 4);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount1(), 3);

    {   // iterator test:
        plask::RectangularFilteredMesh2D::const_iterator it = filteredMesh.begin();

        checkNodeIterator(filteredMesh, it,    0,  1,  plask::vec( 1.0, 4.0));
        checkNodeIterator(filteredMesh, it,    1,  2,  plask::vec( 1.0, 5.0));

        checkNodeIterator(filteredMesh, it,    2,  4,  plask::vec( 2.0, 3.0));
        checkNodeIterator(filteredMesh, it,    3,  5,  plask::vec( 2.0, 4.0));
        checkNodeIterator(filteredMesh, it,    4,  6,  plask::vec( 2.0, 5.0));
        checkNodeIterator(filteredMesh, it,    5,  7,  plask::vec( 2.0, 6.0));

        checkNodeIterator(filteredMesh, it,    6,  8,  plask::vec( 5.0, 3.0));
        checkNodeIterator(filteredMesh, it,    7,  9,  plask::vec( 5.0, 4.0));
        checkNodeIterator(filteredMesh, it,    8, 10,  plask::vec( 5.0, 5.0));
        checkNodeIterator(filteredMesh, it,    9, 11,  plask::vec( 5.0, 6.0));

        checkNodeIterator(filteredMesh, it,   10, 13,  plask::vec(10.0, 4.0));
        checkNodeIterator(filteredMesh, it,   11, 14,  plask::vec(10.0, 5.0));
        checkNodeIterator(filteredMesh, it,   12, 15,  plask::vec(10.0, 6.0));

        checkNodeIterator(filteredMesh, it,   13, 17,  plask::vec(18.0, 4.0));
        checkNodeIterator(filteredMesh, it,   14, 18,  plask::vec(18.0, 5.0));
        checkNodeIterator(filteredMesh, it,   15, 19,  plask::vec(18.0, 6.0));

        BOOST_CHECK(it == filteredMesh.end());
    }

    /*            bottom
     *     0     1     2     3      4
     * 0   |0---2|4---6|8--- |12--- |16            3
     *     |   0 | 1*3 |   6 |    9 |
     * 1  0|1---3|5---7|9--10|13--13|17            4
     *left | 0*1 | 2*4 | 4*7 | 5*10 |    right
     * 2  1|2---4|6---8|10-11|14--14|18            5
     *     |   2 | 3*5 |   8 | 6*11 |
     * 3   |3---5|7---9|11-12|15--15|19            6
     *            top
     *     1     2     5    10     18
     */
    {   // element iterator test:
        plask::RectangularFilteredMesh2D::Elements::const_iterator it = filteredMesh.elements().begin();

        checkElementIterator(filteredMesh, it,   0,  1,   0, 1);

        checkElementIterator(filteredMesh, it,   1,  3,   1, 0);
        checkElementIterator(filteredMesh, it,   2,  4,   1, 1);
        checkElementIterator(filteredMesh, it,   3,  5,   1, 2);

        checkElementIterator(filteredMesh, it,   4,  7,   2, 1);
        checkElementIterator(filteredMesh, it,   5, 10,   3, 1);
        checkElementIterator(filteredMesh, it,   6, 11,   3, 2);

        BOOST_CHECK(it == filteredMesh.elements().end());
    }

    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(0),  0);
    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(2),  1);
    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(3),  2);
    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(4),  3);
    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(7),  4);
    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(10), 5);
    BOOST_CHECK_EQUAL(filteredMesh.getElementIndexFromLowIndex(11), 6);

    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(0),  0);
    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(1),  2);
    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(2),  3);
    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(3),  4);
    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(4),  7);
    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(5), 10);
    BOOST_CHECK_EQUAL(filteredMesh.getElementMeshLowIndex(6), 11);

    checkBoundary(filteredMesh.createLeftBoundary(), {0, 1});
    checkBoundary(filteredMesh.createRightBoundary(), {13, 14, 15});
    checkBoundary(filteredMesh.createBottomBoundary(), {2, 6});
    checkBoundary(filteredMesh.createTopBoundary(), {5, 9, 12, 15});

    {
        plask::DataVector<double> src_data = {
            1, 1,           // 1.0
            2, 2, 2, 2,     // 2.0
            3, 3, 3, 3,     // 5.0
            4, 4, 4,        // 10.0
            5, 5, 5};       // 18.0

        BOOST_CHECK(std::isnan(filteredMesh.interpolateNearestNeighbor(src_data, plask::vec(1.0, 1.0), plask::InterpolationFlags())));
        BOOST_CHECK_EQUAL(filteredMesh.interpolateNearestNeighbor(src_data, plask::vec(1.8, 4.5), plask::InterpolationFlags()), 2.0);
        BOOST_CHECK_EQUAL(filteredMesh.interpolateLinear(src_data, plask::vec(1.5, 4.5), plask::InterpolationFlags()), 1.5);
    }

    {
        plask::DataVector<plask::Vec<2, double>> src_data = {
            plask::vec(1.0, 1.0), plask::vec(1.0, 1.0),           // 1.0
            plask::vec(2.0, 2.0), plask::vec(2.0, 2.0), plask::vec(2.0, 2.0), plask::vec(2.0, 2.0),     // 2.0
            plask::vec(3.0, 3.0), plask::vec(3.0, 3.0), plask::vec(3.0, 3.0), plask::vec(3.0, 3.0),     // 5.0
            plask::vec(4.0, 4.0), plask::vec(4.0, 4.0), plask::vec(4.0, 4.0),        // 10.0
            plask::vec(5.0, 5.0), plask::vec(5.0, 5.0), plask::vec(5.0, 5.0)};       // 18.0
        BOOST_CHECK(plask::isnan(filteredMesh.interpolateNearestNeighbor(src_data, plask::vec(1.0, 1.0), plask::InterpolationFlags())));
        BOOST_CHECK_EQUAL(filteredMesh.interpolateNearestNeighbor(src_data, plask::vec(1.8, 4.5), plask::InterpolationFlags()), plask::vec(2.0, 2.0));
        BOOST_CHECK_EQUAL(filteredMesh.interpolateLinear(src_data, plask::vec(1.5, 4.5), plask::InterpolationFlags()), plask::vec(1.5, 1.5));
    }
}

BOOST_AUTO_TEST_CASE(rectangular_filtered_2D_order10) {
    plask::RectangularFilteredMesh2D filteredMesh = constructMesh(plask::RectangularMesh2D::ORDER_10);
    BOOST_REQUIRE_EQUAL(filteredMesh.size(), 2 + 5 + 5 + 4);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount(), 1 + 4 + 2);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount0(), 4);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount1(), 3);
}

BOOST_AUTO_TEST_CASE(rectangular_filtered_3D) {
    plask::RectangularFilteredMesh3D filteredMesh(
                plask::RectangularMesh3D(
                    plask::make_shared<plask::RegularAxis>(1.0, 4.0, 4),
                    plask::make_shared<plask::RegularAxis>(2.0, 5.0, 4),
                    plask::make_shared<plask::RegularAxis>(3.0, 6.0, 4)),
                [] (const plask::RectangularMesh3D::Element& e) {
                    const std::size_t i0 = e.getIndex0(), i1 = e.getIndex1(), i2 = e.getIndex2();
                    return (i0 == 1 && i1 == 1 && i2 == 1) ||
                           (i0 == 0 && i1 == 1 && i2 == 1) ||
                           (i0 == 0 && i1 == 0 && i2 == 0);
                }
    );
    BOOST_REQUIRE_EQUAL(filteredMesh.size(), 18);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount(), 3);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount0(), 3);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount1(), 3);
    BOOST_REQUIRE_EQUAL(filteredMesh.getElementsCount2(), 3);

    {   // iterator test:
        plask::RectangularFilteredMesh3D::const_iterator it = filteredMesh.begin();

        checkNodeIterator(filteredMesh, it,    0,  0,  plask::vec(1.0, 2.0, 3.0));
        checkNodeIterator(filteredMesh, it,    1,  1,  plask::vec(1.0, 2.0, 4.0));

        checkNodeIterator(filteredMesh, it,    2,  4,  plask::vec(1.0, 3.0, 3.0));
        checkNodeIterator(filteredMesh, it,    3,  5,  plask::vec(1.0, 3.0, 4.0));
        checkNodeIterator(filteredMesh, it,    4,  6,  plask::vec(1.0, 3.0, 5.0));

        checkNodeIterator(filteredMesh, it,    5,  9,  plask::vec(1.0, 4.0, 4.0));
        checkNodeIterator(filteredMesh, it,    6, 10,  plask::vec(1.0, 4.0, 5.0));

        checkNodeIterator(filteredMesh, it,    7, 16,  plask::vec(2.0, 2.0, 3.0));
        checkNodeIterator(filteredMesh, it,    8, 17,  plask::vec(2.0, 2.0, 4.0));

        checkNodeIterator(filteredMesh, it,    9, 20,  plask::vec(2.0, 3.0, 3.0));
        checkNodeIterator(filteredMesh, it,   10, 21,  plask::vec(2.0, 3.0, 4.0));
        checkNodeIterator(filteredMesh, it,   11, 22,  plask::vec(2.0, 3.0, 5.0));

        checkNodeIterator(filteredMesh, it,   12, 25,  plask::vec(2.0, 4.0, 4.0));
        checkNodeIterator(filteredMesh, it,   13, 26,  plask::vec(2.0, 4.0, 5.0));

        checkNodeIterator(filteredMesh, it,   14, 37,  plask::vec(3.0, 3.0, 4.0));
        checkNodeIterator(filteredMesh, it,   15, 38,  plask::vec(3.0, 3.0, 5.0));

        checkNodeIterator(filteredMesh, it,   16, 41,  plask::vec(3.0, 4.0, 4.0));
        checkNodeIterator(filteredMesh, it,   17, 42,  plask::vec(3.0, 4.0, 5.0));

        BOOST_CHECK(it == filteredMesh.end());
    }

    checkBoundary(filteredMesh.createBackBoundary(), { 0, 1, 2, 3, 4, 5, 6 });
    checkBoundary(filteredMesh.createFrontBoundary(), { 14, 15, 16, 17 });
    checkBoundary(filteredMesh.createLeftBoundary(), { 0, 1, 7, 8 });
    checkBoundary(filteredMesh.createRightBoundary(), { 5, 6, 12, 13, 16, 17 });
    checkBoundary(filteredMesh.createBottomBoundary(), { 0, 2, 7, 9 });
    checkBoundary(filteredMesh.createTopBoundary(), { 4, 6, 11, 13, 15, 17 });
}

BOOST_AUTO_TEST_SUITE_END()
