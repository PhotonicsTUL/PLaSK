#include <boost/test/unit_test.hpp>

#include <plask/math.h>

BOOST_AUTO_TEST_SUITE(math) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(parse_complex) {
    //BOOST_CHECK_EQUAL(plask::dcomplex(1.1, 2.2), plask::parse_complex<double>("1.1+2.2j"));
    BOOST_CHECK_EQUAL(plask::dcomplex(3.0, 0.0), plask::parse_complex<double>("3"));
    BOOST_CHECK_EQUAL(plask::dcomplex(0.0, 5.0), plask::parse_complex<double>("5j"));
}

BOOST_AUTO_TEST_SUITE_END()
