#include <boost/test/unit_test.hpp>
#include <plask/vector/tensor3.hpp>

BOOST_AUTO_TEST_SUITE(tensor_axes)

BOOST_AUTO_TEST_CASE(diagonal_accessors) {
    plask::Tensor3<double> tensor(2., 3., 5.);
    const auto& view = tensor;
    BOOST_CHECK_EQUAL(tensor.lon(), 2.);
    BOOST_CHECK_EQUAL(tensor.tran(), 3.);
    tensor.lon() = 7.;
    tensor.tran() = 11.;
    tensor.vert() = 13.;
    BOOST_CHECK_EQUAL(view.lon(), 7.);
    BOOST_CHECK_EQUAL(view.tran(), 11.);
    BOOST_CHECK_EQUAL(view.vert(), 13.);
}

BOOST_AUTO_TEST_SUITE_END()
