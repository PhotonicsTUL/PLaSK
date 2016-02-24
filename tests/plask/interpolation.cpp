#include <boost/test/unit_test.hpp>

#include <plask/mesh/interpolation.h>
#include <plask/mesh/mesh.h>
#include <plask/mesh/rectangular2d.h>
#include <plask/mesh/ordered1d.h>
#include <plask/mesh/rectangular_spline.h>

#include<fstream>

namespace plask {
    struct DummyMesh: public plask::MeshD<2> {
        virtual std::size_t size() const override { return 2; }
        virtual plask::Vec<2, double> at(std::size_t) const override { return plask::vec(0.0, 0.0); }
    };

    template <typename SrcT, typename DstT>
    struct InterpolationAlgorithm<DummyMesh, SrcT, DstT, plask::INTERPOLATION_LINEAR> {
        static LazyData<DstT> interpolate(const shared_ptr<const DummyMesh>&, const DataVector<const SrcT>&,
                                const shared_ptr<const plask::MeshD<DummyMesh::DIM>>& dst_mesh, const InterpolationFlags&) {
            return new ConstValueLazyDataImpl<DstT>(dst_mesh->size(), 11);
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

    template <typename Msh> void print_idx(const Msh& msh, size_t i);
    template <> void print_idx(const plask::shared_ptr<plask::RectangularMesh<2>>& msh, size_t i) {
        std::cerr << msh->index0(i) << msh->index1(i) << " ";
    }
    template <> void print_idx(const plask::shared_ptr<plask::RectangularMesh<3>>& msh, size_t i) {
        std::cerr << msh->index0(i) << msh->index1(i) << msh->index2(i) << " ";
    }

    template <typename SrcMsh, typename DstMsh>
    void test_spline(const DstMsh& dst_mesh, const SrcMsh& src_mesh, const plask::DataVector<const double>& src) {
        plask::DataVector<const double> dst = plask::interpolate(src_mesh, src, dst_mesh, plask::INTERPOLATION_SPLINE);
        for (size_t i = 0; i != src_mesh->size(); ++i)
            print_idx(src_mesh, i);
        std::cerr << " :  ";
        for (size_t i = 0; i != dst_mesh->size(); ++i)
            print_idx(dst_mesh, i);
        std::cerr << "\n";
        // for (size_t i = 0; i != 4; ++i) {
        //     for (size_t j = 0; j != 5; ++j)
        //         std::cerr << plask::format("{:12.9f} ", dst[dst_mesh->index(i,j)]);
        //     std::cerr << "\n";
        // }
        BOOST_CHECK_CLOSE(dst[ 0],  4.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 1],  1.084000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 2],  1.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 3], 12.111111111,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 4], 16.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 5],  6.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 6],  1.626000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 7],  1.233281250,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 8],  6.055555556,  1e-7);
        BOOST_CHECK_CLOSE(dst[ 9],  8.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[10],  6.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[11],  2.888250000,  1e-7);
        BOOST_CHECK_CLOSE(dst[12],  2.417947917,  1e-7);
        BOOST_CHECK_CLOSE(dst[13],  0.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[14],  0.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[15],  4.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[16],  3.122500000,  1e-7);
        BOOST_CHECK_CLOSE(dst[17],  2.872958333,  1e-7);
        BOOST_CHECK_CLOSE(dst[18],  0.000000000,  1e-7);
        BOOST_CHECK_CLOSE(dst[19],  0.000000000,  1e-7);
    }

    BOOST_AUTO_TEST_CASE(spline2d) {
        plask::DataVector<const double> src({4., 8., 4., 1., 2., 3., 1., 0., 0., 16., 0., 0.});
        auto src_mesh = plask::make_shared<plask::RectangularMesh<2>>(
            plask::shared_ptr<plask::OrderedAxis>(new plask::OrderedAxis({0., 1., 2.})),
            plask::shared_ptr<plask::OrderedAxis>(new plask::OrderedAxis({-1., 0., 2., 5.})),
            plask::RectangularMesh<2>::ORDER_10);
        auto dst_mesh = plask::make_shared<plask::RectangularMesh<2>>(
            plask::shared_ptr<plask::OrderedAxis>(new plask::OrderedAxis({0., 0.5, 1.5, 2.})),
            plask::shared_ptr<plask::OrderedAxis>(new plask::OrderedAxis({-1.0, -0.1, 0.1, 4.0, 5.0})),
            plask::RectangularMesh<2>::ORDER_01);
        test_spline(dst_mesh, src_mesh, src);
    }

    BOOST_AUTO_TEST_CASE(spline3d) {
        plask::shared_ptr<plask::OrderedAxis> sa(new plask::OrderedAxis({0., 2.})),
                                              se(new plask::OrderedAxis({-1., 0., 2., 5.})),
                                              si(new plask::OrderedAxis({0., 1., 2.}));
        plask::shared_ptr<plask::OrderedAxis> da(new plask::OrderedAxis({1.})),
                                              de(new plask::OrderedAxis({-1.0, -0.1, 0.1, 4.0, 5.0})),
                                              di(new plask::OrderedAxis({0., 0.5, 1.5, 2.}));
        plask::DataVector<const double> src({4., 8., 4., 1., 2., 3., 1., 0., 0., 16., 0., 0.,
                                             4., 8., 4., 1., 2., 3., 1., 0., 0., 16., 0., 0.});

        test_spline(plask::make_shared<plask::RectangularMesh<3>>(da, di, de),
                    plask::make_shared<plask::RectangularMesh<3>>(sa, si, se, plask::RectangularMesh<3>::ORDER_021),
                    src
                   );

        test_spline(plask::make_shared<plask::RectangularMesh<3>>(di, da, de),
                    plask::make_shared<plask::RectangularMesh<3>>(si, sa, se, plask::RectangularMesh<3>::ORDER_120),
                    src
                   );

        test_spline(plask::make_shared<plask::RectangularMesh<3>>(di, de, da),
                    plask::make_shared<plask::RectangularMesh<3>>(si, se, sa, plask::RectangularMesh<3>::ORDER_210),
                    src
                   );

        test_spline(plask::make_shared<plask::RectangularMesh<3>>(de, di, da, plask::RectangularMesh<3>::ORDER_210),
                    plask::make_shared<plask::RectangularMesh<3>>(se, si, sa, plask::RectangularMesh<3>::ORDER_201),
                    src
                   );
    }

BOOST_AUTO_TEST_SUITE_END()
