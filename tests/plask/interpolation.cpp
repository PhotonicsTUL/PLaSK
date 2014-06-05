#include <boost/test/unit_test.hpp>

#include <plask/mesh/interpolation.h>
#include <plask/mesh/mesh.h>

#include<fstream>

namespace plask {
    struct DummyMesh: public plask::MeshD<2> {
        virtual std::size_t size() const { return 2; }
        virtual plask::Vec<2, double> at(std::size_t) const { return plask::vec(0.0, 0.0); }
    };

    template <typename SrcT, typename DstT>    //for any data type
    struct InterpolationAlgorithm<DummyMesh, SrcT, DstT, plask::INTERPOLATION_LINEAR> {
        static LazyData<DstT> interpolate(const shared_ptr<const DummyMesh>&, const DataVector<const SrcT>&,
                                const shared_ptr<const plask::MeshD<DummyMesh::DIM>>& dst_mesh) {
            return new ConstValueLazyDataImpl<DstT>(11, dst_mesh->size());
        }
    };
}


BOOST_AUTO_TEST_SUITE(interpolation) // MUST be the same as the file name

    BOOST_AUTO_TEST_CASE(interpolation_choice) {
        plask::shared_ptr<const plask::DummyMesh> src_mesh = plask::make_shared<plask::DummyMesh>(),
                                                  dst_mesh = plask::make_shared<plask::DummyMesh>();
        plask::DataVector<int> src_data = {1, 2};

        // Check exceptions
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::__ILLEGAL_INTERPOLATION_METHOD__),
                          plask::CriticalException);
        //BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, (plask::InterpolationMethod)9999),
        //                  plask::CriticalException);
        BOOST_CHECK_THROW(plask::interpolate(src_mesh, src_data, dst_mesh, plask::INTERPOLATION_SPLINE),
                          plask::NotImplemented);

        // Check simple interpolate
        auto ret_data = plask::interpolate<plask::DummyMesh,int>(src_mesh, src_data, dst_mesh, plask::INTERPOLATION_LINEAR);
        BOOST_CHECK_EQUAL(ret_data[0], 11);
    }

BOOST_AUTO_TEST_SUITE_END()
