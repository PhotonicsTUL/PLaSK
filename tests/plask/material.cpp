#include <boost/test/unit_test.hpp>
#include <plask/material/db.hpp>
#include "common/dumb_material.hpp"

BOOST_AUTO_TEST_SUITE(material) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(materialDB) {
        plask::MaterialsDB db;
        BOOST_CHECK_THROW(db.get("Al"), plask::NoSuchMaterial);

        db.add<DumbMaterial>("Al");
        BOOST_CHECK_NO_THROW(db.get("Al"));

        db.remove("Al");
        BOOST_CHECK_THROW(db.get("Al"), plask::NoSuchMaterial);
    }

    BOOST_AUTO_TEST_CASE(defaultMaterialDB) {
        plask::MaterialsDB::loadAllToDefault();
        plask::MaterialsDB& db = plask::MaterialsDB::getDefault();
        BOOST_CHECK_EQUAL(db.get("GaN")->name(), "GaN");
    }

    BOOST_AUTO_TEST_CASE(MaterialInfo_Link_directly_constructed) {
        plask::MaterialInfo::Link link("GaN", plask::MaterialInfo::Mh, "my note");
        BOOST_CHECK_EQUAL(link.className, "GaN");
        BOOST_CHECK_EQUAL(link.property, plask::MaterialInfo::Mh);
        BOOST_CHECK_EQUAL(link.note, "my note");
        BOOST_CHECK_EQUAL(link.str(), "GaN.Mh my note");
    }

    BOOST_AUTO_TEST_CASE(MaterialInfo_Link_constructed_from_string) {
        plask::MaterialInfo::Link link("GaN.Mh my note");
        BOOST_CHECK_EQUAL(link.className, "GaN");
        BOOST_CHECK_EQUAL(link.property, plask::MaterialInfo::Mh);
        BOOST_CHECK_EQUAL(link.note, "my note");
        BOOST_CHECK_EQUAL(link.str(), "GaN.Mh my note");
    }

    BOOST_AUTO_TEST_CASE(MaterialInfo_PropertyInfo) {
        plask::MaterialInfo::PropertyInfo info;
        info.addSource("my source A");
        info.setArgumentRange(plask::MaterialInfo::T, 3.0, 5.0);
        info.addSource("my source B");
        info.addLink(plask::MaterialInfo::Link("GaN", plask::MaterialInfo::Mh, "my note"));
        info.addNote("test note");
        info.setArgumentRange(plask::MaterialInfo::e, 7.5, 8.5);
        BOOST_CHECK_EQUAL(info.get(), "source: my source A\n"
                                      "T range: 3:5\n"
                                      "source: my source B\n"
                                      "see: GaN.Mh my note\n"
                                      "note: test note\n"
                                      "e range: 7.5:8.5");
        BOOST_CHECK_EQUAL(info.getSource(), "my source A\nmy source B");
        BOOST_CHECK_EQUAL(info.getArgumentRange(plask::MaterialInfo::T).first, 3.0);
        BOOST_CHECK_EQUAL(info.getArgumentRange(plask::MaterialInfo::T).second, 5.0);
        BOOST_CHECK_EQUAL(info.getArgumentRange(plask::MaterialInfo::e).first, 7.5);
        BOOST_CHECK_EQUAL(info.getArgumentRange(plask::MaterialInfo::e).second, 8.5);
        BOOST_CHECK(plask::isnan(info.getArgumentRange(plask::MaterialInfo::lam).first));
        BOOST_CHECK(plask::isnan(info.getArgumentRange(plask::MaterialInfo::lam).second));
        auto links = info.getLinks();
        BOOST_REQUIRE_EQUAL(links.size(), 1);
        BOOST_CHECK_EQUAL(links[0].str(), "GaN.Mh my note");
    }

BOOST_AUTO_TEST_SUITE_END()
