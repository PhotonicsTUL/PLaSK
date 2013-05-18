#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectilinear.h>
#include <plask/mesh/boundary_conditions.h>
#include <plask/manager.h>
#include <plask/utils/xml.h>
#include "../common/dumb_material.h"

BOOST_AUTO_TEST_SUITE(boundary_conditions) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(boundary_conditions_rect_simple) {
    plask::BoundaryConditions<plask::RectilinearMesh2D, double> conditions;
    BOOST_CHECK(conditions.empty());
    conditions.add(plask::RectilinearMesh2D::getLeftBoundary(), 1.0);
    conditions.add(plask::RectilinearMesh2D::getRightBoundary(), 2.0);
    BOOST_CHECK_EQUAL(conditions.size(), 2);
    BOOST_CHECK_EQUAL(conditions[0].value, 1.0);

    plask::RectilinearMesh2D mesh;
    mesh.axis0.addPointsLinear(1.0, 3.0, 3);   // 1.0, 2.0, 3.0
    mesh.axis1.addPointsLinear(5.0, 6.0, 2);   // 5.0, 6.0
    //BOOST_CHECK(conditions(mesh).find(0) == conditions(mesh).begin());
}

BOOST_AUTO_TEST_CASE(boundary_conditions_rect_custom) {
    plask::BoundaryConditions<plask::RectilinearMesh2D, double> conditions;
    plask::MaterialsDB materialsDB;
    initDumbMaterialDb(materialsDB);
    plask::Manager manager;
    manager.loadFromXMLString(
                "<plask><geometry><cartesian2d name=\"space\" length=\"1\" axes=\"xy\"><stack>"
                    "<block name=\"top\" dx=\"5\" dy=\"3\" material=\"Al\" />"
                    "<block name=\"bottom\" dx=\"5\" dy=\"3\" material=\"Al\" />"
                "</stack></cartesian2d></geometry></plask>", materialsDB); //repeat=\"2\"
    plask::RectilinearMesh2D::Boundary bottom_b = plask::RectilinearMesh2D::getBottomOfBoundary(manager.getGeometry<plask::GeometryD<2> >("space"), manager.getGeometryObject("bottom"));
    plask::RectilinearMesh2D mesh;
    mesh.axis0.addPointsLinear(1.0, 5.0, 5);   // 1.0, 2.0, 3.0, 4.0, 5.0
    mesh.axis1.addPointsLinear(0.0, 4.0, 5);   // 0.0, 1.0, 2.0, 3.0, 4.0
    plask::RectilinearMesh2D::Boundary::WithMesh wm = bottom_b.get(mesh);
    for (int i = 0; i < 5; ++i) BOOST_CHECK(wm.contains(i));
    for (int i = 5; i < 25; ++i) BOOST_CHECK(!wm.contains(i));
    plask::RectilinearMesh2D::Boundary top_b = plask::RectilinearMesh2D::getTopOfBoundary(manager.getGeometry<plask::GeometryD<2> >("space"), manager.getGeometryObject("top"));
    wm = top_b.get(mesh);
    for (int i = 0; i < 20; ++i) BOOST_CHECK(!wm.contains(i));
    for (int i = 20; i < 25; ++i) BOOST_CHECK(wm.contains(i));
}

BOOST_AUTO_TEST_CASE(boundary_conditions_from_XML) {
    plask::BoundaryConditions<plask::RectilinearMesh2D, double> conditions;
    plask::Manager manager;
    std::string xml_content = "<cond><condition place=\"bottom\" value=\"123\"/><condition place=\"left\" value=\"234\"/></cond>";
    plask::XMLReader reader(new std::stringstream(xml_content));
    BOOST_CHECK_NO_THROW(reader.requireTag("cond"));
    manager.readBoundaryConditions(reader, conditions);
    BOOST_CHECK_EQUAL(conditions.size(), 2);
    BOOST_CHECK_EQUAL(conditions[0].value, 123.0);
    BOOST_CHECK_EQUAL(conditions[1].value, 234.0);
}



BOOST_AUTO_TEST_SUITE_END()
