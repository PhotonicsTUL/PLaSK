#include <boost/test/unit_test.hpp>
#include <plask/data.h>

BOOST_AUTO_TEST_SUITE(data) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(const_datavector) {

        plask::DataVector<int> v = {1, 2, 3, 4};
        plask::DataVector<double> vd;
        plask::DataVector<const int> cv(v);
        plask::DataVector<int> v2;
        plask::DataVector<const int> cv2(v);

        // v = cv;
        // vd = v;

        v2 = plask::const_data_cast<int>(v);
        v2 = plask::const_data_cast<int>(cv);
        cv2 = plask::const_data_cast<const int>(cv);
        cv2 = plask::const_data_cast<const int>(v2);
        cv2 = plask::const_data_cast<int>(cv);
        cv2 = plask::const_data_cast<int>(v2);
        cv2.reset();


        BOOST_CHECK_EQUAL(cv[0], 1);
        BOOST_CHECK_EQUAL(cv[1], 2);
        BOOST_CHECK_EQUAL(cv[2], 3);
        BOOST_CHECK_EQUAL(cv[3], 4);

        BOOST_CHECK_EQUAL(v2[0], 1);
        BOOST_CHECK_EQUAL(v2[1], 2);
        BOOST_CHECK_EQUAL(v2[2], 3);
        BOOST_CHECK_EQUAL(v2[3], 4);

        BOOST_CHECK( !cv.unique() );
        v.reset();
        BOOST_CHECK( !cv.unique() );
        v2.reset();
        BOOST_CHECK( cv.unique() );

    }

BOOST_AUTO_TEST_SUITE_END()
