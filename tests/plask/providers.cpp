#include <boost/test/unit_test.hpp>

#include <plask/provider/provider.h>

BOOST_AUTO_TEST_SUITE(providers) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(single_value) {
    struct OneDouble: public plask::SingleValueProperty<double> {};
    plask::ProviderFor<OneDouble>::WithValue provider;
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

BOOST_AUTO_TEST_CASE(delegate_to_member) {
    struct Obj { double member() { return 1.0; } } obj;
    struct OneDouble: public plask::SingleValueProperty<double> {};
    plask::ProviderFor<OneDouble>::Delegate provider(&obj, &Obj::member);
    plask::ReceiverFor<OneDouble> receiver;
    receiver.setProvider(provider);
    BOOST_CHECK_EQUAL(receiver(), 1.0);
}

BOOST_AUTO_TEST_SUITE_END()
