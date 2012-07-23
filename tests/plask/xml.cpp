#include <boost/test/unit_test.hpp>

#include <plask/utils/xml.h>
#include <plask/utils/xml/writer.h>

#define HEADER "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"

BOOST_AUTO_TEST_SUITE(xml) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(empty_xml) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    writer.writeHeader();
    BOOST_CHECK_EQUAL(ss.str(), HEADER);
}

BOOST_AUTO_TEST_CASE(simple_xml_element) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    writer.writeHeader();
    auto tag = writer.addTag("tag");
    tag.end();
    BOOST_CHECK_EQUAL(ss.str(), HEADER "<tag/>\n");
}

BOOST_AUTO_TEST_CASE(simple_xml_element_with_text) {
    std::stringstream ss;
    {
        plask::XMLWriter writer(ss);
        auto tag = writer.addTag("tag");
        writer.writeText("hello\n");
    }
    BOOST_CHECK_EQUAL(ss.str(), "<tag>\nhello\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(nested_xml_elements_with_text) {

    std::stringstream ss;
    {
      plask::XMLWriter writer(ss);
      auto tag = writer.addTag("tag");
       auto inner = writer.addTag("inner");
        writer.writeText("  hello\n ");
       inner.end();
       writer.addTag("inner");
    } // two tags should be closed automatically
    BOOST_CHECK_EQUAL(ss.str(), "<tag>\n <inner>\n  hello\n </inner>\n <inner/>\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(simple_xml_element_with_attr) {
    std::stringstream ss;
    plask::XMLWriter writer(ss);
    auto tag = writer.addTag("tag");
    tag.attr("name", "Flip").attr("friend", "Flap");
    tag.writeText("hello\n");
    tag.end();
    BOOST_CHECK_EQUAL(ss.str(), "<tag name=\"Flip\" friend=\"Flap\">\nhello\n</tag>\n");
}

BOOST_AUTO_TEST_CASE(xml_element_with_quotes) {
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

BOOST_AUTO_TEST_SUITE_END()
