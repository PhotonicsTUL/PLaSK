#include <boost/test/unit_test.hpp>
#include <plask/filters/filter.h>
#include <plask/mesh/basic.h>

BOOST_AUTO_TEST_SUITE(filters) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(cartesian2D) {
        struct DoubleField: public plask::FieldProperty<double> {};

        plask::Filter<DoubleField, plask::Geometry2DCartesian> filter2D(plask::make_shared<plask::Geometry2DCartesian>());
        filter2D.setDefault(2.0);
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(1.0, 1.0)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });

        //BOOST_CHECK(false);
    }

BOOST_AUTO_TEST_SUITE_END()

