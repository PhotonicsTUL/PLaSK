#include <boost/test/unit_test.hpp>

#include <plask/geometry/geometry.h>

#include "common/dumb_material.h"

struct Leafs2d {
    plask::shared_ptr<plask::Material> dumbMaterial;
    plask::shared_ptr< plask::Block<2> > block_5_3;
    Leafs2d(): dumbMaterial(new DumbMaterial()), block_5_3(new plask::Block<2>(plask::vec(5.0, 3.0), dumbMaterial)) {}
};

BOOST_AUTO_TEST_SUITE(geometry) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(primitives) {
        plask::Rect2d rect(plask::vec(3.0, 2.0), plask::vec(1.0, 1.0));
        rect.fix();
        BOOST_CHECK_EQUAL(rect.lower, plask::vec(1.0, 1.0));
        BOOST_CHECK_EQUAL(rect.upper, plask::vec(3.0, 2.0));
    }

    BOOST_FIXTURE_TEST_CASE(leaf_rectangle, Leafs2d) {
        BOOST_CHECK_EQUAL(block_5_3->getBoundingBox().upper, plask::vec(5.0, 3.0));
        BOOST_CHECK_EQUAL(block_5_3->getBoundingBox().lower, plask::vec(0.0, 0.0));
        BOOST_CHECK_EQUAL(block_5_3->getMaterial(plask::vec(4.0, 2.0)), dumbMaterial);
        BOOST_CHECK(block_5_3->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
        BOOST_CHECK_NO_THROW(block_5_3->validate());
    }

    BOOST_FIXTURE_TEST_CASE(translate2d, Leafs2d) {
        plask::shared_ptr<plask::Translation<2>> translation(new plask::Translation<2>(block_5_3, plask::vec(10.0, 20.0)));    //should be in [10, 20] - [15, 23]
        BOOST_CHECK_EQUAL(translation->getBoundingBox(), plask::Rect2d(plask::vec(10, 20), plask::vec(15, 23)));
        BOOST_CHECK_EQUAL(translation->getMaterial(plask::vec(12.0, 22.0)), dumbMaterial);
        BOOST_CHECK(translation->getMaterial(plask::vec(4.0, 22.0)) == nullptr);
    }
    
    BOOST_FIXTURE_TEST_CASE(translationContainer2d, Leafs2d) {
        plask::shared_ptr<plask::TranslationContainer<2>> container(new plask::TranslationContainer<2>);
        container->add(block_5_3);
        container->add(block_5_3, plask::vec(3.0, 3.0));
        BOOST_CHECK_EQUAL(container->getBoundingBox(), plask::Rect2d(plask::vec(0.0, 0.0), plask::vec(8.0, 6.0)));
        BOOST_CHECK_EQUAL(container->getMaterial(plask::vec(6.0, 6.0)), dumbMaterial);
        BOOST_CHECK(container->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
    }

    BOOST_FIXTURE_TEST_CASE(multistack2d, Leafs2d) {
        plask::shared_ptr<plask::MultiStackContainer<2>> multistack(new plask::MultiStackContainer<2>(5, 10.0));
        multistack->add(block_5_3);
        multistack->add(block_5_3);
        //5 * 2 childs = 10 elements, each have size 5x3, should be in [0, 10] - [5, 40]
        BOOST_CHECK_EQUAL(multistack->getBoundingBox(), plask::Rect2d(plask::vec(0.0, 10.0), plask::vec(5.0, 40.0)));
        BOOST_CHECK_EQUAL(multistack->getMaterial(plask::vec(4.0, 39.0)), dumbMaterial);
        BOOST_CHECK(multistack->getMaterial(plask::vec(4.0, 41.0)) == nullptr);
    }

    BOOST_AUTO_TEST_CASE(manager_loading) {
        plask::MaterialsDB materialsDB;
        initDumbMaterialDb(materialsDB);
        plask::GeometryManager manager;
        manager.loadFromXMLString("<geometry axis=\"xy\"><stack2d repeat=\"2\"><child><block2d name=\"block\" x=\"4\" y=\"2\" material=\"Dumb\" /></child><ref name=\"block\" /></stack2d></geometry>", materialsDB);
        //BOOST_CHECK_EQUAL(manager.elements.size(), 3);
        BOOST_CHECK(manager.getElement("block") != nullptr);
        BOOST_CHECK(manager.getElement("notexist") == nullptr);
    }

BOOST_AUTO_TEST_SUITE_END()
