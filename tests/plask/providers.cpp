#include <boost/test/unit_test.hpp>

#include <plask/provider/provider.h>

BOOST_AUTO_TEST_SUITE(providers_and_recivers)

BOOST_AUTO_TEST_CASE(single_value) {
	struct OneDouble: public plask::SingleValueProperty<double> {};
	plask::ProviderFor<OneDouble> provider;
	plask::ReceiverFor<OneDouble> receiver;
	
	BOOST_CHECK_THROW(receiver(), plask::NoProvider);
	BOOST_CHECK(receiver.changed);
	BOOST_CHECK(receiver.getProvider() == nullptr);
	receiver.setProvider(provider);
	BOOST_CHECK(receiver.changed);
	BOOST_CHECK_EQUAL(receiver.getProvider(), &provider);
	
	provider() = 1.0;
	BOOST_CHECK_EQUAL(provider(), 1.0);
	BOOST_CHECK(receiver.changed);
	BOOST_CHECK_EQUAL(receiver(), 1.0);
	BOOST_CHECK(!receiver.changed);

	receiver.setProvider(0);
	BOOST_CHECK(receiver.changed);
	BOOST_CHECK_THROW(receiver(), plask::NoProvider);
}

BOOST_AUTO_TEST_SUITE_END()
