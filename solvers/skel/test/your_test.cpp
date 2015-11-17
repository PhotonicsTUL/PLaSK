#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Your solver test"
#include <boost/test/unit_test.hpp>

// This is needed because of a bug in Boost linkage in the newest Ubuntu
namespace boost { namespace unit_test { namespace ut_detail {
BOOST_TEST_DECL std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}

#include "../your_solver.h"
using namespace plask;

BOOST_AUTO_TEST_SUITE(your_solver)

BOOST_AUTO_TEST_CASE(your_test)
{
}


BOOST_AUTO_TEST_SUITE_END()
