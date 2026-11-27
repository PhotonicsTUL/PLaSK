#include <boost/test/unit_test.hpp>
#include <plask/data.hpp>

namespace {
struct ConstructionFailure {};

struct Element {
    static int live;
    static int remaining;
    int value;
    explicit Element(int value = 0): value(value) {
        if (remaining == 0) throw ConstructionFailure{};
        if (remaining > 0) --remaining;
        ++live;
    }
    Element(const Element& src): Element(src.value) {}
    Element& operator=(const Element&) = delete;
    ~Element() { --live; }
};
int Element::live = 0;
int Element::remaining = -1;
}

BOOST_AUTO_TEST_SUITE(data_owned_storage)

BOOST_AUTO_TEST_CASE(construct_in_raw_storage) {
    {
        Element initial(4);
        plask::DataVector<const Element> values(3, initial);
        BOOST_CHECK_EQUAL(Element::live, 4);
        values.reset(2u, values[0]);
        BOOST_CHECK_EQUAL(values[1].value, 4);
        BOOST_CHECK_EQUAL(Element::live, 3);

        const plask::DataVector<int> integers = {2, 3};
        plask::DataVector<Element> converted(integers);
        BOOST_CHECK_EQUAL(converted[1].value, 3);
        plask::DataVector<Element> list{6, 7};
        BOOST_CHECK_EQUAL(list[0].value, 6);
        list.reset(converted.begin(), converted.end());
        BOOST_CHECK_EQUAL(list[0].value, 2);
    }
    BOOST_CHECK_EQUAL(Element::live, 0);
}

BOOST_AUTO_TEST_CASE(unwind_partial_construction) {
    {
        Element value(7);
        plask::DataVector<Element> original(1, value);
        const auto* data = original.data();
        Element::remaining = 2;
        BOOST_CHECK_THROW(original.reset(5), ConstructionFailure);
        BOOST_CHECK_EQUAL(Element::live, 2);
        BOOST_CHECK(original.data() == data);
        BOOST_CHECK_EQUAL(original.size(), 1u);

        Element::remaining = 2;
        BOOST_CHECK_THROW(original.reset(5u, value), ConstructionFailure);
        Element::remaining = -1;
        BOOST_CHECK_EQUAL(Element::live, 2);
        BOOST_CHECK(original.data() == data);
        BOOST_CHECK_EQUAL(original[0].value, 7);
    }
    BOOST_CHECK_EQUAL(Element::live, 0);
}

BOOST_AUTO_TEST_SUITE_END()
