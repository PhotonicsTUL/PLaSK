#include <boost/test/unit_test.hpp>

#include <plask/vector/2d.h>
#include <plask/vector/3d.h>

BOOST_AUTO_TEST_SUITE(vectors)

BOOST_AUTO_TEST_CASE(Vector2d) {
    plask::Vector2d<double> v_d(1.0, 2.0);
    BOOST_CHECK(v_d.x == 1.0 && v_d.y == 2.0);
   
    //plask::Vector2d<float> v_f(1.0, 2.0);
}

BOOST_AUTO_TEST_CASE(Vector3d) {
    plask::Vector3d<double> v_d(1.0, 2.0, 3.0);
    BOOST_CHECK(v_d.x == 1.0 && v_d.y == 2.0 && v_d.z == 3.0);
   
    //plask::Vector2d<float> v_f(1.0, 2.0);
}


BOOST_AUTO_TEST_SUITE_END()