#include <boost/test/unit_test.hpp>
#include <plask/memalloc.hpp>
#include <cstdint>

BOOST_AUTO_TEST_SUITE(aligned_reallocation)

BOOST_AUTO_TEST_CASE(preserve_prefix) {
    for (bool supply_old_size : {false, true}) {
        auto* data = plask::aligned_malloc<unsigned char>(32);
        for (unsigned char i = 0; i != 32; ++i) data[i] = i;
        data = static_cast<unsigned char*>(plask::aligned_realloc(data, 4096, supply_old_size ? 32 : 0));
        BOOST_CHECK_EQUAL(reinterpret_cast<std::uintptr_t>(data) % 16, 0u);
        for (unsigned char i = 0; i != 32; ++i) BOOST_CHECK_EQUAL(data[i], i);
        data = static_cast<unsigned char*>(plask::aligned_realloc(data, 19, supply_old_size ? 4096 : 0));
        BOOST_CHECK_EQUAL(reinterpret_cast<std::uintptr_t>(data) % 16, 0u);
        for (unsigned char i = 0; i != 19; ++i) BOOST_CHECK_EQUAL(data[i], i);
        plask::aligned_free(data);
    }
}

BOOST_AUTO_TEST_SUITE_END()
