#include <boost/test/unit_test.hpp>
#include <plask/memalloc.hpp>

BOOST_AUTO_TEST_SUITE(aligned_size)

BOOST_AUTO_TEST_CASE(element_count_overflow) {
    struct UnexpectedConstruction {};
    struct Element {
        double value;
        Element() { throw UnexpectedConstruction{}; }
    };
    const auto max_count = (std::numeric_limits<std::size_t>::max)() / sizeof(Element);
    plask::aligned_allocator<Element> allocator;
    BOOST_CHECK_EQUAL(allocator.max_size(), max_count);
    BOOST_CHECK_THROW(allocator.allocate(max_count + 1), std::bad_alloc);
    BOOST_CHECK_THROW(plask::aligned_new_array<Element>(max_count + 1), std::bad_alloc);
}

#if !PLASK_MALLOC_ALIGNED && !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
BOOST_AUTO_TEST_CASE(alignment_padding_overflow) {
    const auto max_bytes = (std::numeric_limits<std::size_t>::max)();
    BOOST_CHECK_THROW(plask::aligned_malloc(max_bytes), std::bad_alloc);
    plask::aligned_unique_ptr<char> block(plask::aligned_malloc<char>());
    *block = 42;
    BOOST_CHECK_THROW(plask::aligned_realloc(block.get(), max_bytes), std::bad_alloc);
    BOOST_CHECK_EQUAL(*block, 42);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
