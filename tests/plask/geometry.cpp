#include <boost/test/unit_test.hpp>

#include <plask/geometry/geometry.h>
#include <plask/geometry/stack.h>

#include "common/dumb_material.h"

struct Leafs2D {
    plask::shared_ptr<plask::Material> dumbMaterial;
    plask::shared_ptr<plask::Block<2>> block_5_3;
    plask::shared_ptr<plask::Block<2>> block_5_4;
    Leafs2D(): dumbMaterial(new DumbMaterial()),
        block_5_3(new plask::Block<2>(plask::vec(5.0, 3.0), dumbMaterial)),
        block_5_4(new plask::Block<2>(plask::vec(5.0, 4.0), dumbMaterial)) {}
};

void test_multi_stack(plask::shared_ptr<plask::MultiStackContainer<2>> multistack, plask::PathHints& p) {
    // 5 * 2 children = 10 elements, each have size 5x3, should be in [0, 10] - [5, 40]
    BOOST_CHECK_EQUAL(multistack->getBoundingBox(), plask::Box2D(plask::vec(0.0, 10.0), plask::vec(5.0, 40.0)));
    BOOST_CHECK(multistack->getMaterial(plask::vec(4.0, 39.0)) != nullptr);
    BOOST_CHECK(multistack->getMaterial(plask::vec(4.0, 41.0)) == nullptr);
    BOOST_CHECK_EQUAL(multistack->getLeafsBoundingBoxes(p).size(), 5);
    BOOST_CHECK_EQUAL(multistack->getLeafs().size(), 10);
    {
        auto leafs = multistack->getLeafs();
        BOOST_REQUIRE_EQUAL(leafs.size(), 10);
        for (int i = 0; i < 10; ++i) BOOST_CHECK_MESSAGE(leafs[i] != nullptr, i << "-th leaf (from getLeafs) is nullptr");
    }
    {
        std::vector<plask::Box2D> bb = multistack->getLeafsBoundingBoxes();
        BOOST_REQUIRE_EQUAL(bb.size(), 10);
        for (int i = 0; i < 10; ++i) BOOST_CHECK_EQUAL(bb[i], plask::Box2D(plask::vec(0.0, 10.0 + i*3), plask::vec(5.0, 10.0 + 3.0 + i*3)));
    }
    {
        std::vector< plask::Vec<2, double> > leafsTran = multistack->getLeafsPositions();
        BOOST_REQUIRE_EQUAL(leafsTran.size(), 10);
        for (int i = 0; i < 10; ++i) {
            BOOST_CHECK_EQUAL(leafsTran[i], plask::vec(0.0, 10.0 + i*3));
        }
    }
}

BOOST_AUTO_TEST_SUITE(geometry) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(primitives) {
        plask::Box2D rect(plask::vec(3.0, 2.0), plask::vec(1.0, 1.0));
        rect.fix();
        BOOST_CHECK_EQUAL(rect.lower, plask::vec(1.0, 1.0));
        BOOST_CHECK_EQUAL(rect.upper, plask::vec(3.0, 2.0));
    }

    BOOST_FIXTURE_TEST_CASE(leaf_block2D, Leafs2D) {
        BOOST_CHECK_EQUAL(block_5_3->getBoundingBox().upper, plask::vec(5.0, 3.0));
        BOOST_CHECK_EQUAL(block_5_3->getBoundingBox().lower, plask::vec(0.0, 0.0));
        BOOST_CHECK_EQUAL(block_5_3->getMaterial(plask::vec(4.0, 2.0)), dumbMaterial);
        BOOST_CHECK(block_5_3->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
        BOOST_CHECK_NO_THROW(block_5_3->validate());
    }

    BOOST_FIXTURE_TEST_CASE(translate2D, Leafs2D) {
        plask::shared_ptr<plask::Translation<2>> translation(new plask::Translation<2>(block_5_3, plask::vec(10.0, 20.0)));    //should be in [10, 20] - [15, 23]
        BOOST_CHECK_EQUAL(translation->getBoundingBox(), plask::Box2D(plask::vec(10, 20), plask::vec(15, 23)));
        BOOST_CHECK_EQUAL(translation->getMaterial(plask::vec(12.0, 22.0)), dumbMaterial);
        BOOST_CHECK(translation->getMaterial(plask::vec(4.0, 22.0)) == nullptr);
    }

    BOOST_FIXTURE_TEST_CASE(translationContainer2D, Leafs2D) {
        plask::shared_ptr<plask::TranslationContainer<2>> container(new plask::TranslationContainer<2>);
        container->add(block_5_3);
        container->add(block_5_3, plask::vec(3.0, 3.0));
        BOOST_CHECK_EQUAL(container->getBoundingBox(), plask::Box2D(plask::vec(0.0, 0.0), plask::vec(8.0, 6.0)));
        BOOST_CHECK_EQUAL(container->getMaterial(plask::vec(6.0, 6.0)), dumbMaterial);
        BOOST_CHECK(container->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
    }

    BOOST_FIXTURE_TEST_CASE(multistack2D, Leafs2D) {
        plask::shared_ptr<plask::MultiStackContainer<2>> multistack(new plask::MultiStackContainer<2>(5, 10.0));
        multistack->add(block_5_3, plask::align::Tran(0.0));
        plask::PathHints p; p += multistack->add(block_5_3, plask::align::Tran(0.0));
        test_multi_stack(multistack, p);
        BOOST_CHECK_EQUAL(multistack->getMaterial(plask::vec(4.0, 39.0)), dumbMaterial);
    }

    BOOST_FIXTURE_TEST_CASE(stack2D, Leafs2D) {
        plask::shared_ptr<plask::StackContainer<2>> stack(new plask::StackContainer<2>(0.0));
        for (int i = 0; i < 3; ++i) {
            stack->add(block_5_3, plask::align::Tran(0.0));
            stack->add(block_5_4, plask::align::Tran(0.0));
        }   //3x(3+4)=21
        BOOST_CHECK_EQUAL(stack->getRealChildrenCount(), 6);
        BOOST_CHECK_EQUAL(stack->getBoundingBox(), plask::Box2D(0.0, 0.0, 5.0, 21.0));
        stack->removeAt(5); //remove one 5x4 block
        BOOST_CHECK_EQUAL(stack->getRealChildrenCount(), 5);
        BOOST_CHECK_EQUAL(stack->getBoundingBox(), plask::Box2D(0.0, 0.0, 5.0, 21.0 - 4.0));
        stack->remove(block_5_3);   //remove all, 3, 5x3 blocks, on stack stay 2 5x4 blocks
        BOOST_CHECK_EQUAL(stack->getRealChildrenCount(), 2);
        BOOST_CHECK_EQUAL(stack->getBoundingBox(), plask::Box2D(0.0, 0.0, 5.0, 8.0));
    }

    BOOST_AUTO_TEST_CASE(manager_loading) {
        plask::MaterialsDB materialsDB;
        initDumbMaterialDb(materialsDB);
        plask::GeometryManager manager;
        manager.loadFromXMLString(
                    "<geometry><cartesian2d name=\"space\" length=\"1\" axes=\"xy\"><stack repeat=\"5\" from=\"10\" name=\"multistack\">"
                    "<child x=\"0\"><block name=\"block_5_3\" x=\"5\" y=\"3\" material=\"Al\" /></child>"
                    "<child x=\"0\" path=\"p\"><ref name=\"block_5_3\" /></child>"
                    "</stack></cartesian2d></geometry>", materialsDB);
        //BOOST_CHECK_EQUAL(manager.elements.size(), 3);
        BOOST_CHECK(manager.getElement("block_5_3") != nullptr);
        BOOST_CHECK(manager.getElement("notexist") == nullptr);
        test_multi_stack(manager.getElement<plask::MultiStackContainer<2>>("multistack"), manager.requirePathHints("p"));
    }

    BOOST_AUTO_TEST_CASE(path_from_vector) {
        plask::shared_ptr<plask::StackContainer<2>> stack1( new plask::StackContainer<2> );
        plask::shared_ptr<plask::StackContainer<2>> stack2( new plask::StackContainer<2> );
        plask::shared_ptr<plask::Rectangle> element( new plask::Rectangle(plask::vec(1,2), plask::MaterialsDB::getDefault().get("GaN")) );
        stack2->add(element);
        stack1->add(stack2);

        std::vector<plask::shared_ptr<const plask::GeometryElement>> list = {stack2, stack1};
        plask::Path path(list);
        path += element;
    }

    BOOST_AUTO_TEST_CASE(empty_containters) {
        plask::shared_ptr<plask::StackContainer<2>> stack1(new plask::StackContainer<2>);
        plask::shared_ptr<plask::StackContainer<2>> stack2(new plask::StackContainer<2>);
        stack1->add(stack2);
    }

BOOST_AUTO_TEST_SUITE_END()
