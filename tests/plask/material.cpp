#include <boost/test/unit_test.hpp>

#include "common/dumb_material.h"

BOOST_AUTO_TEST_SUITE(material) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(materialDB) {
        plask::MaterialsDB db;
        BOOST_CHECK_THROW(db.get("Al"), plask::NoSuchMaterial);

        DumbMaterial material;
    }

BOOST_AUTO_TEST_SUITE_END()