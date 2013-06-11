#include <boost/test/unit_test.hpp>
#include <plask/filters/filter.h>
#include <plask/mesh/basic.h>
#include <plask/geometry/geometry.h>
#include "common/dumb_material.h"

struct TestEnvGeom2D {

    plask::shared_ptr<plask::Block<2>> block11;
    plask::shared_ptr<plask::TranslationContainer<2>> container;
    plask::shared_ptr<plask::Extrusion> extrusion;

    TestEnvGeom2D() {
        block11 = plask::make_shared<plask::Block<2>>(plask::vec(1.0, 1.0), plask::make_shared<DumbMaterial>());
        container = plask::make_shared<plask::TranslationContainer<2>>();
        extrusion = plask::make_shared<plask::Extrusion>(container, 10.0);
        container->add(block11, plask::vec(1.0, 1.0));
        container->add(block11, plask::vec(2.0, 2.0));
    }

};

struct TestEnvGeom3D {

    plask::shared_ptr<plask::Block<3>> block111;
    plask::shared_ptr<plask::Block<2>> block11_in_extr;
    plask::shared_ptr<plask::Extrusion> extr211;
    plask::shared_ptr<plask::TranslationContainer<3>> container;

    TestEnvGeom3D() {
        block111 = plask::make_shared<plask::Block<3>>(plask::vec(1.0, 1.0, 1.0), plask::make_shared<DumbMaterial>());
        block11_in_extr = plask::make_shared<plask::Block<2>>(plask::vec(1.0, 1.0), plask::make_shared<DumbMaterial>());
        extr211 = plask::make_shared<plask::Extrusion>(block11_in_extr, 2.0);
        container = plask::make_shared<plask::TranslationContainer<3>>();
        container->add(block111, plask::vec(1.0, 1.0, 1.0));    // to [2, 2, 2]
        container->add(block111, plask::vec(2.0, 2.0, 2.0));    // to [3, 3, 3]
        container->add(extr211, plask::vec(3.0, 3.0, 3.0));     // to [5, 4, 4]
    }

};

BOOST_AUTO_TEST_SUITE(filters) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(cartesian2D) {
        struct DoubleField: public plask::FieldProperty<double> {};

        TestEnvGeom2D g;

        plask::Filter<DoubleField, plask::Geometry2DCartesian> filter2D(plask::make_shared<plask::Geometry2DCartesian>(g.extrusion));

        filter2D.setDefault(1.0);
        filter2D.appendInner(g.block11) = 2.0;
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(0.5, 0.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 1.0 });
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(1.5, 1.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(2.5, 2.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });
        filter2D.setOuter(g.extrusion) = 3.0;
        BOOST_CHECK_EQUAL(filter2D.out(plask::toMesh(plask::vec(0.5, 0.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 3.0 });
    }

    BOOST_AUTO_TEST_CASE(cartesian3D) {
        struct DoubleField: public plask::FieldProperty<double> {};

        TestEnvGeom3D e;

        plask::Filter<DoubleField, plask::Geometry3D> filter3D(plask::make_shared<plask::Geometry3D>(e.container));
        filter3D.setDefault(1.0);
        filter3D.appendInner(e.block111) = 2.0;
        filter3D.appendInner2D(e.extr211) = 3.0;
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(0.5, 0.5, 0.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 1.0 });
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(1.5, 1.5, 1.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(2.5, 2.5, 2.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 2.0 });
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(4.5, 3.5, 3.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<double>{ 3.0 });
    }

    BOOST_AUTO_TEST_CASE(cartesian3Dvectorfields) {
        struct VectorField: public plask::VectorFieldProperty<> {};

        TestEnvGeom3D e;

        plask::Filter<VectorField, plask::Geometry3D> filter3D(plask::make_shared<plask::Geometry3D>(e.container));
        filter3D.setDefault(plask::vec(1.0, 1.0, 1.0));
        filter3D.appendInner(e.block111) = plask::vec(2.0, 2.0, 2.0);
        filter3D.appendInner2D(e.extr211) = plask::vec(3.0, 3.0);

        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(0.5, 0.5, 0.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<plask::Vec<3>>{ plask::vec(1.0, 1.0, 1.0) });
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(1.5, 1.5, 1.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<plask::Vec<3>>{ plask::vec(2.0, 2.0, 2.0) });
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(2.5, 2.5, 2.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<plask::Vec<3>>{ plask::vec(2.0, 2.0, 2.0) });
        BOOST_CHECK_EQUAL(filter3D.out(plask::toMesh(plask::vec(4.5, 3.5, 3.5)), plask::DEFAULT_INTERPOLATION), plask::DataVector<plask::Vec<3>>{ plask::vec(0.0, 3.0, 3.0) });
    }

BOOST_AUTO_TEST_SUITE_END()

