#include <boost/test/unit_test.hpp>
#include <plask/data.h>

BOOST_AUTO_TEST_SUITE(datavector) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(const_datavector) {

        plask::DataVector<int> v = {1, 2, 3, 4};
        plask::DataVector<double> vd;
        plask::DataVector<const int> cv(v);

        // cv = v;
        // vd = v;

        BOOST_CHECK_EQUAL(cv[0], 1);
        BOOST_CHECK_EQUAL(cv[1], 2);
        BOOST_CHECK_EQUAL(cv[2], 3);
        BOOST_CHECK_EQUAL(cv[3], 4);

        BOOST_CHECK( !cv.unique() );
        v.reset();
        BOOST_CHECK( cv.unique() );

    }

BOOST_AUTO_TEST_SUITE_END()
