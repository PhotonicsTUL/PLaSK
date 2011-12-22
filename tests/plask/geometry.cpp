#include <boost/test/unit_test.hpp>

#include <plask/geometry/leaf.h>
#include <plask/geometry/transform.h>

#include <plask/material/material.h>
#include <config.h>

struct DumpMaterial: public plask::Material {
    virtual std::string getName() const { return "Dump"; }
};

struct Leafs2d {
    shared_ptr<plask::Material> dumpMaterial;
    plask::Block<2> block_5_3;
    Leafs2d(): dumpMaterial(new DumpMaterial()), block_5_3(plask::vec(5.0, 3.0), dumpMaterial) {}
};

BOOST_AUTO_TEST_SUITE(geometry) // MUST be the same as the file name

    BOOST_FIXTURE_TEST_CASE(leaf_box2d, Leafs2d) {
        BOOST_CHECK_EQUAL(block_5_3.getBoundingBox().upper, plask::vec(5.0, 3.0));
        BOOST_CHECK_EQUAL(block_5_3.getBoundingBox().lower, plask::vec(0.0, 0.0));
        BOOST_CHECK_EQUAL(block_5_3.getMaterial(plask::vec(4.0, 2.0)), dumpMaterial);
        BOOST_CHECK(block_5_3.getMaterial(plask::vec(6.0, 2.0)) == nullptr);
        BOOST_CHECK_NO_THROW(block_5_3.validate());
    }

    BOOST_FIXTURE_TEST_CASE(translate2d, Leafs2d) {
        plask::Translation<2> translation(block_5_3, plask::vec(10.0, 20.0));    //should be in [10, 20] - [15, 23]
        BOOST_CHECK_EQUAL(translation.getBoundingBox(), plask::Rect2d(plask::vec(10, 20), plask::vec(15, 23)));
        BOOST_CHECK_EQUAL(translation.getMaterial(plask::vec(12.0, 22.0)), dumpMaterial);
        BOOST_CHECK(translation.getMaterial(plask::vec(4.0, 22.0)) == nullptr);
    }


BOOST_AUTO_TEST_SUITE_END()
