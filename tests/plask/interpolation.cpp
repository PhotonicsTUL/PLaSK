#include <boost/test/unit_test.hpp>

#include <plask/mesh/interpolation.h>
#include <plask/mesh/mesh.h>

#include<fstream>

namespace plask {
    struct DummyMesh: public plask::Mesh<2> {
        virtual std::size_t size() const { return 1; }
        virtual plask::Mesh<2>::Iterator begin() const { plask::Mesh<2>::Iterator i; return i; }
        virtual plask::Mesh<2>::Iterator end() const {  plask::Mesh<2>::Iterator i; return i; }
    };

    template <typename DataT>    //for any data type
    struct InterpolationAlgorithm<DummyMesh, DataT, plask::LINEAR> {
        static void interpolate(DummyMesh& src_mesh, const std::vector<DataT>& src_vec,
                                const plask::Mesh<DummyMesh::dim>& dst_mesh, std::vector<DataT>& dst_vec) {
            dst_vec[0] = src_vec[0] + 10;
        }
    };
}


BOOST_AUTO_TEST_SUITE(interpolation) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(interpolation_choice) {
        plask::DummyMesh src_mesh, dst_mesh;
        plask::shared_ptr<std::vector<int>> src_data {new std::vector<int> {1, 2}};

        // Check exceptions
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::__ILLEGAL_INTERPOLATION_METHOD__),
                          plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, (plask::InterpolationMethod)9999),
                          plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::SPLINE),
                          plask::NotImplemented);

        // Check simple interpolate
        plask::shared_ptr<const std::vector<int>>ret_data = plask::interpolate<plask::DummyMesh,int>(src_mesh, src_data, dst_mesh, plask::LINEAR);
        BOOST_CHECK_EQUAL((*ret_data)[0], 11);
    }

BOOST_AUTO_TEST_SUITE_END()
