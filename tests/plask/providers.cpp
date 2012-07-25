#include <type_traits> // for remove_reference
#include <boost/test/unit_test.hpp>

#include <plask/provider/provider.h>
#include <plask/provider/temperature.h>
#include <plask/geometry/space.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/regular.h>

BOOST_AUTO_TEST_SUITE(providers) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(single_value) {
    struct OneDouble: public plask::SingleValueProperty<double> {};
    plask::ProviderFor<OneDouble>::WithDefaultValue provider;
    plask::ReceiverFor<OneDouble> receiver;

    BOOST_CHECK_THROW(receiver(), plask::NoProvider);
    BOOST_CHECK(receiver.changed);
    BOOST_CHECK(receiver.getProvider() == nullptr);
    receiver.setProvider(provider);
    BOOST_CHECK(receiver.changed);
    BOOST_CHECK_EQUAL(receiver.getProvider(), &provider);

    provider = 1.0;
    BOOST_CHECK_EQUAL(provider(), 1.0);
    BOOST_CHECK(receiver.changed);
    BOOST_CHECK_EQUAL(receiver(), 1.0);
    BOOST_CHECK(!receiver.changed);

    receiver.setProvider(0);
    BOOST_CHECK(receiver.changed);
    BOOST_CHECK_THROW(receiver(), plask::NoProvider);

    receiver.setConstValue(3.0);
    BOOST_CHECK_EQUAL(receiver(), 3.0);

    receiver = 2.0;
    BOOST_CHECK_EQUAL(receiver(), 2.0);

    plask::ProviderFor<OneDouble>::WithValue providerOpt;
    BOOST_CHECK(!providerOpt.hasValue());
    receiver.setProvider(providerOpt);
    BOOST_CHECK_THROW(receiver(), plask::NoValue);
    providerOpt = 4.0;
    BOOST_CHECK_EQUAL(receiver(), 4.0);
    providerOpt.invalidate();
    BOOST_CHECK_THROW(receiver(), plask::NoValue);
}

BOOST_AUTO_TEST_CASE(delegate_to_member) {
    struct Obj { double member() { return 1.0; } } obj;
    struct OneDouble: public plask::SingleValueProperty<double> {};
    plask::ProviderFor<OneDouble>::Delegate provider(&obj, &Obj::member);
    plask::ReceiverFor<OneDouble> receiver;
    receiver << provider;
    BOOST_CHECK_EQUAL(receiver(), 1.0);
}

BOOST_AUTO_TEST_CASE(polymorphic_receivers) {
    struct OneDouble: public plask::SingleValueProperty<double> {};
    plask::ProviderFor<OneDouble>::WithDefaultValue provider;
    plask::Receiver<plask::SingleValueProvider<double>> receiver;

    receiver.setProvider(provider);
    provider = 1.0;
    BOOST_CHECK_EQUAL(receiver(), 1.0);
}

BOOST_AUTO_TEST_CASE(attach_datavector)
{
    auto mesh1 = plask::make_shared<plask::RegularMesh2D>(plask::RegularMesh1D(0., 4., 3), plask::RegularMesh1D(0., 20., 3));

    auto mesh2 = plask::make_shared<std::remove_reference<decltype(*mesh1)>::type>(mesh1->getMidpointsMesh());

    plask::DataVector<double> data(9);
    data[0] = 100.; data[1] = 100.; data[2] = 100.;
    data[3] = 300.; data[4] = 300.; data[5] = 300.;
    data[6] = 500.; data[7] = 500.; data[8] = 500.;

    plask::ReceiverFor<plask::Temperature, plask::Geometry2DCartesian> receiver;
    receiver.setValue(data, mesh1);

    BOOST_CHECK_EQUAL(receiver(*mesh1).data(), data.data());

    auto result2 = receiver(*mesh2);
    BOOST_CHECK_EQUAL(result2[0], 200.);
    BOOST_CHECK_EQUAL(result2[1], 200.);
    BOOST_CHECK_EQUAL(result2[2], 400.);
    BOOST_CHECK_EQUAL(result2[3], 400.);

    BOOST_CHECK_EQUAL(data.unique(), false);
    mesh1->setIterationOrder(plask::RegularMesh2D::TRANSPOSED_ORDER);
    BOOST_CHECK_EQUAL(data.unique(), true);
}

BOOST_AUTO_TEST_SUITE_END()
