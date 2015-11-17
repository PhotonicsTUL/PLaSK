#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE "plask Unit Tests"
#include <boost/test/unit_test.hpp>

namespace boost { namespace unit_test { namespace ut_detail {
std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}

//aboves declare int main(...);
