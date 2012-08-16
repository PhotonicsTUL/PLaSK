#include <boost/test/unit_test.hpp>
#include <plask/material/db.h>
#include "common/dumb_material.h"

BOOST_AUTO_TEST_SUITE(material) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(materialDB) {
        plask::MaterialsDB db;
        BOOST_CHECK_THROW(db.get("Al"), plask::NoSuchMaterial);

        db.add<DumbMaterial>("Al");
        BOOST_CHECK_NO_THROW(db.get("Al"));

        db.remove<DumbMaterial>("Al");
        BOOST_CHECK_THROW(db.get("Al"), plask::NoSuchMaterial);
    }

    BOOST_AUTO_TEST_CASE(defaultMaterialDB) {
//         TODO load materials dynamically
//         plask::initDefaultMaterials();
        plask::MaterialsDB& db = plask::MaterialsDB::getDefault();
        BOOST_CHECK_EQUAL(db.get("GaN")->name(), "GaN");
    }

BOOST_AUTO_TEST_SUITE_END()
