#undef _GLIBCXX_DEBUG

#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE "plask Unit Tests"
#include <boost/test/unit_test.hpp>

#if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
namespace boost { namespace unit_test { namespace ut_detail {
std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}
#endif

//aboves declare int main(...);
