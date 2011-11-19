#include <boost/test/unit_test.hpp>

#include <plask/provider/provider.h>

BOOST_AUTO_TEST_SUITE(providers_and_recivers)

BOOST_AUTO_TEST_CASE(single_value) {
	struct OneDouble: public plask::SingleValueProperty<double> {};
	plask::Provider<OneDouble> provider;
	plask::Reciver<OneDouble> reciver;
	
	BOOST_CHECK_THROW(reciver(), plask::NoProvider);
	BOOST_CHECK(reciver.changed);
	BOOST_CHECK(reciver.getProvider() == nullptr);
	reciver.setProvider(provider);
	BOOST_CHECK(reciver.changed);
	BOOST_CHECK_EQUAL(reciver.getProvider(), &provider);
	
	provider() = 1.0;
	BOOST_CHECK_EQUAL(provider(), 1.0);
	BOOST_CHECK(reciver.changed);
	BOOST_CHECK_EQUAL(reciver(), 1.0);
	BOOST_CHECK(!reciver.changed);
	
	reciver.setProvider(0);
	BOOST_CHECK(reciver.changed);
	BOOST_CHECK_THROW(reciver(), plask::NoProvider);
}

BOOST_AUTO_TEST_SUITE_END()
