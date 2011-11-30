#include <boost/test/unit_test.hpp>

#include <plask/vector/2d.h>
#include <plask/vector/3d.h>

BOOST_AUTO_TEST_SUITE(vectors)

BOOST_AUTO_TEST_CASE(Vector2d) {
    plask::Vec2<double> v_d(1.0, 2.0);
    BOOST_CHECK(v_d.c0 == 1.0 && v_d.c1 == 2.0);
   
    //plask::Vector2d<float> v_f(1.0, 2.0);
}

BOOST_AUTO_TEST_CASE(Vector3d) {
    plask::Vec3<double> v_d(1.0, 2.0, 3.0);
    BOOST_CHECK(v_d.c0 == 1.0 && v_d.c1 == 2.0 && v_d.c2 == 3.0);
   
    //plask::Vector2d<float> v_f(1.0, 2.0);
}


BOOST_AUTO_TEST_SUITE_END()