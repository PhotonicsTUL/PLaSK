#include <boost/test/unit_test.hpp>

#include <plask/math.h>

BOOST_AUTO_TEST_SUITE(math) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(parse_complex) {
    BOOST_CHECK_EQUAL(plask::dcomplex(1.1, 2.2), plask::parse_complex<double>("1.1+2.2j"));
    BOOST_CHECK_EQUAL(plask::dcomplex(6.6, 7.7), plask::parse_complex<double>("(6.6+7.7j)"));
    BOOST_CHECK_EQUAL(plask::dcomplex(5e-2, 1e2), plask::parse_complex<double>(" 5e-2 + 1e2 j "));
    BOOST_CHECK_EQUAL(plask::dcomplex(3.0, 0.0), plask::parse_complex<double>("3"));
    BOOST_CHECK_EQUAL(plask::dcomplex(0.0, 5.0), plask::parse_complex<double>("5j"));
    BOOST_CHECK_EQUAL(plask::dcomplex(3.0, 4.0), plask::parse_complex<double>("(3, 4)"));
    BOOST_CHECK_THROW(plask::parse_complex<double>("2+2"), plask::IllFormatedComplex);
    BOOST_CHECK_THROW(plask::parse_complex<double>("not a complex"), plask::IllFormatedComplex);
    BOOST_CHECK_THROW(plask::parse_complex<double>("2+3j !"), plask::IllFormatedComplex);
}

BOOST_AUTO_TEST_SUITE_END()
