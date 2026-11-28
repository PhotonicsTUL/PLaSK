#include <boost/test/unit_test.hpp>
#include <plask/data.hpp>

namespace {
struct CopyFailure {};
struct Deleter {
    bool& fail_copy;
    int& calls;
    Deleter(bool& fail_copy, int& calls): fail_copy(fail_copy), calls(calls) {}
    Deleter(const Deleter& src): fail_copy(src.fail_copy), calls(src.calls) {
        if (fail_copy) throw CopyFailure{};
    }
    void operator()(void*) const { ++calls; }
};
}

BOOST_AUTO_TEST_SUITE(data_deleter_reset)

BOOST_AUTO_TEST_CASE(throwing_deleter_copy) {
    int old_value = 10, new_value = 20;
    int old_calls = 0, new_calls = 0;
    bool fail_copy = false;
    const std::function<void(void*)> deleter = Deleter(fail_copy, new_calls);
    plask::DataVector<int> values(&old_value, 1, [&](void*) { ++old_calls; });
    fail_copy = true;
    BOOST_CHECK_THROW(values.reset(&new_value, 1, deleter), CopyFailure);
    BOOST_CHECK_EQUAL(old_calls, 0);
    BOOST_CHECK_EQUAL(new_calls, 0);
    BOOST_CHECK(values.data() == &old_value);
    fail_copy = false;
    values.reset(&new_value, 1, deleter);
    BOOST_CHECK_EQUAL(old_calls, 1);
    BOOST_CHECK(values.data() == &new_value);
    values.reset();
    BOOST_CHECK_EQUAL(new_calls, 1);
}

BOOST_AUTO_TEST_SUITE_END()
