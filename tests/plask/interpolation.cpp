#include <boost/test/unit_test.hpp>

#include <plask/mesh/interpolation.h>
#include <plask/mesh/mesh.h>

namespace plask {
    struct DummyMesh: public plask::Mesh {
        virtual std::size_t size() const { return 1; }
        virtual plask::Mesh::Iterator begin() { plask::Mesh::Iterator i; return i; }
        virtual plask::Mesh::Iterator end() {  plask::Mesh::Iterator i; return i; }
    };

    template <typename DataT>    //for any data type
    struct InterpolationAlgorithm<DummyMesh, DataT, plask::LINEAR> {
        static void interpolate(DummyMesh& src_mesh, const std::vector<DataT>& src_vec, const plask::Mesh& dst_mesh, std::vector<DataT>& dst_vec)
        throw (plask::NotImplemented) {
            dst_vec.resize(dst_mesh.size());
            dst_vec[0] = 12;
        }
    };
}

BOOST_AUTO_TEST_SUITE(interpolation_alorithms)

BOOST_AUTO_TEST_CASE(interpolation_choice) {

        plask::DummyMesh src_mesh, dst_mesh;
        std::shared_ptr<const std::vector<int>> src_data;

        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::__ILLEGAL_INTERPOLATION_METHOD__), plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, (plask::InterpolationMethod)9999), plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::SPLINE), plask::NotImplemented);

        std::shared_ptr<const std::vector<int>> dst_data = plask::interpolate(src_mesh, src_data, dst_mesh, plask::LINEAR);
        BOOST_CHECK_EQUAL((*dst_data)[0], 12.);
}

BOOST_AUTO_TEST_SUITE_END()
