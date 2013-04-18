#include <boost/test/unit_test.hpp>
#include <plask/filters/filter.h>
#include <plask/mesh/basic.h>
#include <plask/geometry/geometry.h>
#include "common/dumb_material.h"

BOOST_AUTO_TEST_SUITE(filters) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(cartesian2D) {
        struct DoubleField: public plask::FieldProperty<double> {};

        plask::shared_ptr<plask::Material> dump_material = plask::make_shared<DumbMaterial>();
        plask::shared_ptr<plask::Block<2>> block11 = plask::make_shared<plask::Block<2>>(plask::vec(1.0, 1.0), dump_material);
        plask::shared_ptr<plask::TranslationContainer<2>> container = plask::make_shared<plask::TranslationContainer<2>>();
        plask::shared_ptr<plask::Extrusion> extrusion = plask::make_shared<plask::Extrusion>(container, 10.0);

        container->add(block11, plask::vec(1.0, 1.0));
        container->add(block11, plask::vec(2.0, 2.0));

        plask::Filter<DoubleField, plask::Geometry2DCartesian> filter2D(plask::make_shared<plask::Geometry2DCartesian>(extrusion));

        filter2D.setDefault(1.0);
        filter2D.appendInner(block11) = 2.0;
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(0.5, 0.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 1.0 });
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(1.5, 1.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(2.5, 2.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });

        filter2D.setOuter(extrusion) = 3.0;
        //BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(0.5, 0.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 3.0 });
    }

BOOST_AUTO_TEST_SUITE_END()

