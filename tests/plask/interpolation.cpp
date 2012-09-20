#include <boost/test/unit_test.hpp>

#include <plask/mesh/interpolation.h>
#include <plask/mesh/mesh.h>

#include<fstream>

namespace plask {
    struct DummyMesh: public plask::MeshD<2> {
        virtual std::size_t size() const { return 1; }
        virtual plask::Vec<2, double> at(std::size_t) const { return plask::vec(0.0, 0.0); }
    };

    template <typename DataT>    //for any data type
    struct InterpolationAlgorithm<DummyMesh, DataT, plask::INTERPOLATION_LINEAR> {
        static void interpolate(const DummyMesh& src_mesh, const DataVector<const DataT>& src_vec,
                                const plask::MeshD<DummyMesh::DIM>& dst_mesh, DataVector<DataT>& dst_vec) {
            dst_vec[0] = src_vec[0] + 10;
        }
    };
}


BOOST_AUTO_TEST_SUITE(interpolation) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(interpolation_choice) {
        plask::DummyMesh src_mesh, dst_mesh;
        plask::DataVector<int> src_data = {1, 2};

        // Check exceptions
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::__ILLEGAL_INTERPOLATION_METHOD__),
                          plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, (plask::InterpolationMethod)9999),
                          plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::INTERPOLATION_SPLINE),
                          plask::NotImplemented);

        // Check simple interpolate
        auto ret_data = plask::interpolate<plask::DummyMesh,int>(src_mesh, src_data, dst_mesh, plask::INTERPOLATION_LINEAR);
        BOOST_CHECK_EQUAL(ret_data[0], 11);
    }

BOOST_AUTO_TEST_SUITE_END()
