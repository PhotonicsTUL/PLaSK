#include <boost/test/unit_test.hpp>

#include <plask/mesh/rectangular.h>
#include <plask/mesh/boundary_conditions.h>
#include <plask/manager.h>
#include <plask/utils/xml.h>
#include "../common/dumb_material.h"

BOOST_AUTO_TEST_SUITE(boundary_conditions) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(boundary_conditions_rect_simple) {
    plask::BoundaryConditions<plask::RectangularMesh<2>::Boundary, double> conditions;
    BOOST_CHECK(conditions.empty());
    conditions.add(plask::RectangularMesh<2>::getLeftBoundary(), 1.0);
    conditions.add(plask::RectangularMesh<2>::getRightBoundary(), 2.0);
    BOOST_CHECK_EQUAL(conditions.size(), 2);
    BOOST_CHECK_EQUAL(conditions[0].value, 1.0);

    /*plask::RectangularMesh<2> mesh;
    mesh.axis0.addPointsLinear(1.0, 3.0, 3);   // 1.0, 2.0, 3.0
    mesh.axis1.addPointsLinear(5.0, 6.0, 2);   // 5.0, 6.0
    BOOST_CHECK(conditions(mesh).find(0) == conditions(mesh).begin()); //this line was commented out in trunk*/
}

BOOST_AUTO_TEST_CASE(boundary_conditions_rect_custom) {
    plask::BoundaryConditions<plask::RectangularMesh<2>::Boundary, double> conditions;
    plask::MaterialsDB materialsDB;
    initDumbMaterialDb(materialsDB);
    plask::Manager manager;
    manager.loadFromXMLString(
                "<plask><geometry><cartesian2d name=\"space\" length=\"1\" axes=\"xy\"><stack>"
                    "<block name=\"top\" dx=\"5\" dy=\"3\" material=\"Al\" />"
                    "<block name=\"bottom\" dx=\"5\" dy=\"3\" material=\"Al\" />"
                "</stack></cartesian2d></geometry></plask>", materialsDB); //repeat=\"2\"
    plask::RectangularMesh<2>::Boundary bottom_b = plask::RectangularMesh<2>::getBottomOfBoundary(manager.getGeometryObject("bottom"));
    auto axis0 = plask::make_shared<plask::OrderedAxis>();
    auto axis1 = plask::make_shared<plask::OrderedAxis>();
    plask::RectangularMesh<2> mesh(axis0, axis1, plask::RectangularMesh<2>::ORDER_10);
    axis0->addPointsLinear(1.0, 5.0, 5);   // 1.0, 2.0, 3.0, 4.0, 5.0
    axis1->addPointsLinear(0.0, 4.0, 5);   // 0.0, 1.0, 2.0, 3.0, 4.0
    plask::BoundaryNodeSet wm = bottom_b.get(mesh, manager.getGeometry<plask::GeometryD<2> >("space"));
    for (int i = 0; i < 5; ++i) BOOST_CHECK(wm.contains(i));
    for (int i = 5; i < 25; ++i) BOOST_CHECK(!wm.contains(i));
    plask::RectangularMesh<2>::Boundary top_b = plask::RectangularMesh<2>::getTopOfBoundary(manager.getGeometryObject("top"));
    wm = top_b.get(mesh, manager.getGeometry<plask::GeometryD<2> >("space"));
    for (int i = 0; i < 20; ++i) BOOST_CHECK(!wm.contains(i));
    for (int i = 20; i < 25; ++i) BOOST_CHECK(wm.contains(i));
}

BOOST_AUTO_TEST_CASE(boundary_conditions_from_XML) {
    plask::BoundaryConditions<plask::RectangularMesh<2>::Boundary, double> conditions;
    plask::Manager manager;
    std::string xml_content = "<cond><condition place=\"bottom\" value=\"123\"/><condition place=\"left\" value=\"234\"/></cond>";
    plask::XMLReader reader(std::unique_ptr<std::istream>(new std::stringstream(xml_content)));
    BOOST_CHECK_NO_THROW(reader.requireTag("cond"));
    manager.readBoundaryConditions(reader, conditions);
    BOOST_CHECK_EQUAL(conditions.size(), 2);
    BOOST_CHECK_EQUAL(conditions[0].value, 123.0);
    BOOST_CHECK_EQUAL(conditions[1].value, 234.0);
}



BOOST_AUTO_TEST_SUITE_END()
