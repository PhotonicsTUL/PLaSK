#include <boost/test/unit_test.hpp>
#include <plask/memalloc.hpp>

namespace {

struct ConstructionFailure {};

struct Counted {
    int& live;
    int index;
    bool& reverse_order;

    Counted(int& live, int fail_at, bool& reverse_order)
        : live(live), index(live), reverse_order(reverse_order) {
        if (index == fail_at) throw ConstructionFailure{};
        ++live;
    }

    ~Counted() {
        --live;
        if (index != live) reverse_order = false;
    }
};

} // namespace

BOOST_AUTO_TEST_SUITE(aligned_construction)

BOOST_AUTO_TEST_CASE(throwing_constructors) {
    int live = 0;
    bool reverse_order = true;
    BOOST_CHECK_THROW(plask::aligned_new<Counted>(live, 0, reverse_order), ConstructionFailure);
    BOOST_CHECK_THROW(plask::aligned_new_array<Counted>(5, live, 3, reverse_order), ConstructionFailure);
    BOOST_CHECK_EQUAL(live, 0);
    BOOST_CHECK(reverse_order);

    auto* data = plask::aligned_new_array<Counted>(2, live, 2, reverse_order);
    BOOST_CHECK_EQUAL(live, 2);
    plask::aligned_delete_array(2, data);
    BOOST_CHECK_EQUAL(live, 0);
    BOOST_CHECK(reverse_order);
}

BOOST_AUTO_TEST_SUITE_END()
