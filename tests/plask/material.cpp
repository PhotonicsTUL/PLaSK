#include <boost/test/unit_test.hpp>

#include "common/dump_material.h"

BOOST_AUTO_TEST_SUITE(material) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(materialDB) {
        plask::MaterialsDB db;
        BOOST_CHECK_THROW(db.get("Al"), plask::NoSuchMaterial);
        
        DumpMaterial material;
    }

BOOST_AUTO_TEST_SUITE_END()