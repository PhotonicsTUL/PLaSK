#include <boost/test/unit_test.hpp>

#include "plask/vec.hpp"

BOOST_AUTO_TEST_SUITE(vectors) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(Vector2D) {
    plask::Vec<2,double> v_d(1.0, 2.0);
    BOOST_CHECK(v_d.c0 == 1.0 && v_d.c1 == 2.0);

    //plask::Vector2D<float> v_f(1.0, 2.0);
}

BOOST_AUTO_TEST_CASE(Vector3D) {
    plask::Vec<3,double> v_d(1.0, 2.0, 3.0);
    BOOST_CHECK(v_d.c0 == 1.0 && v_d.c1 == 2.0 && v_d.c2 == 3.0);

    //plask::Vector2D<float> v_f(1.0, 2.0);
}

BOOST_AUTO_TEST_CASE(VectorOperations) {
    plask::Vec<2,double> a(1., 2.), b(10., 20.);
    BOOST_CHECK_EQUAL(plask::dot(a, b), 50.);
}

// BOOST_AUTO_TEST_CASE(space_conversion) {
//     plask::Vec<2,double> v2 = plask::SpaceXY::local(0., 1., 2.);
//     BOOST_CHECK(v2 == plask::Vec<2,double>(1.,2.));
//
//     plask::Vec<2,double> v3 = plask::SpaceXY::local(plask::Vec<3,double>(0., 1., 2.));
//     BOOST_CHECK(v3 == plask::Vec<2,double>(1.,2.));
// }


BOOST_AUTO_TEST_SUITE_END()
