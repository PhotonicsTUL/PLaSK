#include <boost/test/unit_test.hpp>

#include <plask/utils/xml.h>
#include <plask/utils/xml/writer.h>
#include <plask/utils/xml/reader.h>

#include <plask/mesh/rectangular.h>

#define HEADER "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"

BOOST_AUTO_TEST_SUITE(xml) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(xml_read) {
    plask::XMLReader r(new std::stringstream("<tag a1=\"1\" a2=\"2.0\">3</tag>"));
    r.stringInterpreter.set([] (const std::string&) { return 0; });  //all int will be parsed as 0
    BOOST_CHECK(r.read());
    BOOST_CHECK(r.getNodeType() == plask::XMLReader::NODE_ELEMENT);
    BOOST_CHECK_EQUAL(r.getTagName(), "tag");
    BOOST_CHECK_EQUAL(r.getAttribute<int>("a1", 2), 0);
    BOOST_CHECK_EQUAL(r.getAttribute<double>("a2", 1.0), 2.0);
    BOOST_CHECK_EQUAL(r.getAttribute<double>("a3", 1.0), 1.0);
    BOOST_CHECK(r.read());  //content
    BOOST_CHECK(r.getNodeType() == plask::XMLReader::NODE_TEXT);
    BOOST_CHECK_EQUAL(r.getTextContent<int>(), 0);
    BOOST_CHECK_EQUAL(r.getTextContent<double>(), 3.0);
    BOOST_CHECK(r.read());  //tag end
    BOOST_CHECK(r.getNodeType() == plask::XMLReader::NODE_ELEMENT_END);
    //BOOST_CHECK(!r.read()); //and nothing more, end
}

BOOST_AUTO_TEST_CASE(empty_xml) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    writer.writeHeader();
    BOOST_CHECK_EQUAL(ss.str(), HEADER);
}

BOOST_AUTO_TEST_CASE(simple_xml_object) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    writer.writeHeader();
    auto tag = writer.addTag("tag");
    tag.end();
    BOOST_CHECK_EQUAL(ss.str(), HEADER "<tag/>\n");
}

BOOST_AUTO_TEST_CASE(simple_xml_object_with_text) {
    std::stringstream ss;
    {
        plask::XMLWriter writer(ss);
        auto tag = writer.addTag("tag");
        writer.writeText("hello\n");
    }
    BOOST_CHECK_EQUAL(ss.str(), "<tag>\nhello\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(nested_xml_objects_with_text) {

    std::stringstream ss;
    {
      plask::XMLWriter writer(ss);
      auto tag = writer.addTag("tag");
       auto inner = writer.addTag("inner");
        writer.indent();
        writer.writeText("hello\n");
       inner.end();
       writer.addTag("inner");
    } // two tags should be closed automatically
    BOOST_CHECK_EQUAL(ss.str(), "<tag>\n  <inner>\n    hello\n  </inner>\n  <inner/>\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(simple_xml_object_with_attr) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    auto tag = writer.addTag("tag");
    tag.attr("name", "Flip").attr("friend", "Flap");
    tag.writeText("hello\n");
    tag.end();
    BOOST_CHECK_EQUAL(ss.str(), "<tag name=\"Flip\" friend=\"Flap\">\nhello\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(xml_object_with_quotes) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    {
        auto tag = writer.addTag("tag");
        tag.attr("name", "Flip").attr("friend", "\"Flap\"");
        writer.writeText("<hello>\n");
    }
    BOOST_CHECK_EQUAL(ss.str(), "<tag name=\"Flip\" friend=\"&quot;Flap&quot;\">\n&lt;hello&gt;\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(numeric_attributes) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    {
        auto tag = writer.addTag("tag");
        tag.attr("age", 20).attr("height", 2.5).attr("single", false);
    }
    BOOST_CHECK_EQUAL(ss.str(), "<tag age=\"20\" height=\"2.5\" single=\"0\"/>\n");
}

BOOST_AUTO_TEST_CASE(cdata) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    auto tag = writer.addTag("tag");
    writer.writeCDATA("<hello>");
    tag.end();
    BOOST_CHECK_EQUAL(ss.str(), "<tag>\n<![CDATA[<hello>]]></tag>\n");
}

BOOST_AUTO_TEST_CASE(mesh) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    writer.writeHeader();
    auto grids = writer.addTag("grids");

    auto mesh2 = plask::RectangularMesh<2>(plask::make_shared<plask::RegularAxis>(1,5,3), plask::make_shared<plask::RegularAxis>(10, 40, 4));
    mesh2.writeXML(writer.addTag("mesh").attr("name", "reg"));

    auto mesh3 = plask::RectangularMesh<3>(
                plask::shared_ptr<plask::RectilinearAxis>(new plask::RectilinearAxis{1, 2, 3}),
                plask::shared_ptr<plask::RectilinearAxis>(new plask::RectilinearAxis{20, 50}),
                plask::shared_ptr<plask::RectilinearAxis>(new plask::RectilinearAxis{10})
         );
    mesh3.writeXML(writer.addTag("mesh").attr("name", "rec"));

    grids.end();

    BOOST_CHECK_EQUAL(ss.str(), HEADER
        "<grids>\n"
        "  <mesh name=\"reg\" type=\"rectangular2d\">\n"
        "    <axis0 type=\"regular1d\" start=\"1\" stop=\"5\" num=\"3\"/>\n"
        "    <axis1 type=\"regular1d\" start=\"10\" stop=\"40\" num=\"4\"/>\n"
        "  </mesh>\n"
        "  <mesh name=\"rec\" type=\"rectangular3d\">\n"
        "    <axis0 type=\"rectilinear1d\">\n"
        "      1 2 3 \n"
        "    </axis0>\n"
        "    <axis1 type=\"rectilinear1d\">\n"
        "      20 50 \n"
        "    </axis1>\n"
        "    <axis2 type=\"rectilinear1d\">\n"
        "      10 \n"
        "    </axis2>\n"
        "  </mesh>\n"
        "</grids>\n"
    );
}

BOOST_AUTO_TEST_SUITE_END()
