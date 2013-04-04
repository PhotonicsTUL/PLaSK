#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "plask.optical.effective solver test"
#include <boost/test/unit_test.hpp>

#include <plask/plask.hpp>
#include "../eim.h"
#include "../patterson.h"
using namespace plask;
using namespace plask::solvers::effective;

BOOST_AUTO_TEST_SUITE(eimtest)

    BOOST_AUTO_TEST_CASE(integral) {
        double err = 1e-15;
        dcomplex integral = patterson([](dcomplex x){ return exp(x*x); }, -1, 3, err); // 1446.00777463862
        BOOST_CHECK( abs(integral - 1446.00777463862) < 1e-11 );
    }

BOOST_AUTO_TEST_SUITE_END()
