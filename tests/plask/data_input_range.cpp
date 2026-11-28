#include <boost/test/unit_test.hpp>
#include <plask/data.hpp>
#include <sstream>

BOOST_AUTO_TEST_SUITE(data_input_range)

BOOST_AUTO_TEST_CASE(single_pass_reset) {
    std::istringstream input("3 5 8");
    using Iterator = std::istream_iterator<int>;
    plask::DataVector<const int> values;
    values.reset(Iterator(input), Iterator());
    const int expected[] = {3, 5, 8};
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(), std::begin(expected), std::end(expected));

    values.reset(Iterator(input), Iterator());
    BOOST_CHECK_EQUAL(values.size(), 0u);
}

BOOST_AUTO_TEST_CASE(forward_range_and_fill) {
    plask::DataVector<int> values = {2, 4, 6, 8};
    values.reset(values.begin() + 1, values.end() - 1);
    const int expected[] = {4, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(values.begin(), values.end(), std::begin(expected), std::end(expected));
    values.reset(2, 7);
    BOOST_REQUIRE_EQUAL(values.size(), 2u);
    BOOST_CHECK_EQUAL(values[0], 7);
    BOOST_CHECK_EQUAL(values[1], 7);
}

BOOST_AUTO_TEST_SUITE_END()
