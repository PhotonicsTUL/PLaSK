#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE "electrical/fem2d Tests"

#include <boost/test/unit_test.hpp>


#include "../femT.h"

BOOST_AUTO_TEST_SUITE(mesh)

BOOST_AUTO_TEST_CASE(Vector2D) {
    plask::Vec<2,double> v_d(1.0, 2.0);
    BOOST_CHECK(v_d.c0 == 1.0 && v_d.c1 == 2.0);

    //plask::Vector2D<float> v_f(1.0, 2.0);
}

BOOST_AUTO_TEST_SUITE_END()

