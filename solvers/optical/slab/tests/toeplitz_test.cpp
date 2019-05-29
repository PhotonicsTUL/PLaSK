#undef _GLIBCXX_DEBUG

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Toeplitz test"
#include <boost/test/unit_test.hpp>

#if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
namespace boost { namespace unit_test { namespace ut_detail {
std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}
#endif

#define CHECK_CLOSE_COLLECTION(aa, bb, tolerance) { \
    using std::distance; \
    using std::begin; \
    using std::end; \
    auto a = begin(aa), ae = end(aa); \
    auto b = begin(bb); \
    BOOST_REQUIRE_EQUAL(distance(a, ae), distance(b, end(bb))); \
    double total_error = 0.;\
    for(; a != ae; ++a, ++b) total_error += abs2(*a - *b); \
    BOOST_CHECK_SMALL(total_error, double(distance(begin(aa), ae)) * tolerance); \
}

#include "../fourier/toeplitz.h"
using namespace plask;
using namespace plask::optical::slab;

BOOST_AUTO_TEST_SUITE(toeplitz)

BOOST_AUTO_TEST_CASE(simple)
{
    // Test forward transform
    DataVector<dcomplex> T = { 6., 3., 2., 1., 1., 2., 4. };
    cmatrix X(4, 2, { 24., 35., 42., 38., -3.,  8.,  4., -1. });
    cmatrix R(4, 2, { 1., 2., 3., 4., -2., 2., 1., -1. });

    ToeplitzLevinson(T, X);
    CHECK_CLOSE_COLLECTION(X, R, 1e-14)
}

BOOST_AUTO_TEST_SUITE_END()
