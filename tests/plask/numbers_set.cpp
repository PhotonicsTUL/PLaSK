#include <boost/test/unit_test.hpp>

#include <plask/utils/numbers_set.h>

typedef plask::CompressedSetOfNumbers<std::size_t> Set;

void check_2_567_9(const Set& set) {    // test if the set is a proper set of numbers {2, 5, 6, 7, 9}
    BOOST_CHECK_EQUAL(set.segmentsCount(), 3);
    BOOST_CHECK(!set.empty());
    BOOST_CHECK_EQUAL(set.size(), 5);
    BOOST_CHECK_EQUAL(set.at(0), 2);
    BOOST_CHECK_EQUAL(set[0], 2);
    BOOST_CHECK_EQUAL(set.at(1), 5);
    BOOST_CHECK_EQUAL(set.at(2), 6);
    BOOST_CHECK_EQUAL(set.at(3), 7);
    BOOST_CHECK_EQUAL(set.at(4), 9);
    BOOST_CHECK_EQUAL(set[4], 9);
    BOOST_CHECK_THROW(set.at(5), plask::OutOfBoundsException);
    BOOST_CHECK_EQUAL(set.indexOf(0), Set::NOT_INCLUDED);
    BOOST_CHECK_EQUAL(set.indexOf(1), Set::NOT_INCLUDED);
    BOOST_CHECK_EQUAL(set.indexOf(2), 0);
    BOOST_CHECK_EQUAL(set.indexOf(3), Set::NOT_INCLUDED);
    BOOST_CHECK_EQUAL(set.indexOf(4), Set::NOT_INCLUDED);
    BOOST_CHECK_EQUAL(set.indexOf(5), 1);
    BOOST_CHECK_EQUAL(set.indexOf(6), 2);
    BOOST_CHECK_EQUAL(set.indexOf(7), 3);
    BOOST_CHECK_EQUAL(set.indexOf(8), Set::NOT_INCLUDED);
    BOOST_CHECK_EQUAL(set.indexOf(9), 4);
    BOOST_CHECK_EQUAL(set.indexOf(10), Set::NOT_INCLUDED);
    auto it = set.begin();  // iterator test
    BOOST_CHECK(it != set.end());
    BOOST_CHECK_EQUAL(*it++, 2);
    BOOST_CHECK(it != set.end());
    BOOST_CHECK_EQUAL(*it++, 5);
    BOOST_CHECK_EQUAL(*it++, 6);
    BOOST_CHECK_EQUAL(*it++, 7);
    BOOST_CHECK(it != set.end());
    BOOST_CHECK_EQUAL(*it++, 9);
    BOOST_CHECK(it == set.end());
}

BOOST_AUTO_TEST_SUITE(numbers_set) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(empty) {
    Set set;
    BOOST_CHECK_EQUAL(set.segmentsCount(), 0);
    BOOST_CHECK(set.empty());
    BOOST_CHECK_EQUAL(set.size(), 0);
    set.push_back(0);
    BOOST_CHECK(!set.empty());
    BOOST_CHECK_EQUAL(set.size(), 1);
    set.clear();
    BOOST_CHECK(set.empty());
    BOOST_CHECK_EQUAL(set.size(), 0);
}

BOOST_AUTO_TEST_CASE(push_back) {
    Set set;
    set.push_back(2);
    set.push_back(5);
    set.push_back(6);
    set.push_back(7);
    set.push_back(9);
    check_2_567_9(set);
}

BOOST_AUTO_TEST_CASE(insert1) {
    Set set;
    set.insert(7);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 1);
    set.insert(2);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 2);
    set.insert(5);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 3);
    set.insert(9);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 4);
    set.insert(6);
    check_2_567_9(set);
}

BOOST_AUTO_TEST_CASE(insert2) {
    Set set;
    set.insert(6);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 1);
    set.insert(5);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 1);
    set.insert(7);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 1);
    set.insert(2);
    BOOST_CHECK_EQUAL(set.segmentsCount(), 2);
    set.insert(9);
    check_2_567_9(set);
}

BOOST_AUTO_TEST_CASE(intersection) {
    BOOST_CHECK_EQUAL(Set({1, 2, 3}).intersection(Set({2, 3, 4})), Set({2, 3}));
    BOOST_CHECK_EQUAL(Set({1, 2, 5, 6, 7, 9, 10}).intersection(Set({2, 3, 4, 5, 8, 9})), Set({2, 5, 9}));
    BOOST_CHECK_EQUAL(Set({6, 7, 9, 10}).intersection(Set({2, 3, 4, 5})), Set());
    BOOST_CHECK_EQUAL(Set({1, 2, 3}).intersection(Set()), Set());
    BOOST_CHECK_EQUAL(Set().intersection(Set()), Set());
}


BOOST_AUTO_TEST_SUITE_END()
