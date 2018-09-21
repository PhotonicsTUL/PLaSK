#include "rectangular_filtered_spline.h"
#include "hyman.h"
#include "../exceptions.h"
#include "../math.h"


namespace plask {

#define NOT_INCLUDED CompressedSetOfNumbers<std::size_t>::NOT_INCLUDED


template <typename DstT, typename SrcT>
SplineFilteredRect2DLazyDataImpl<DstT, SrcT>::SplineFilteredRect2DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh,
                                                                               const DataVector<const SrcT>& src_vec,
                                                                               const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                               const InterpolationFlags& flags):
    InterpolatedLazyDataImpl<DstT, RectangularFilteredMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
    diff0(src_mesh->size()), diff1(src_mesh->size()) {}

template <typename DstT, typename SrcT>
DstT SplineFilteredRect2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<2> p;
    size_t i0_lo, i0_hi, i1_lo, i1_hi;

    if (!this->src_mesh->prepareInterpolation(this->dst_mesh->at(index), p, i0_lo, i0_hi, i1_lo, i1_hi, this->flags))
        return NaNfor<decltype(this->src_vec[0])>();

    double left = this->src_mesh->fullMesh.getAxis0()->at(i0_lo), right = this->src_mesh->fullMesh.getAxis0()->at(i0_hi),
           bottom = this->src_mesh->fullMesh.getAxis1()->at(i1_lo), top = this->src_mesh->fullMesh.getAxis1()->at(i1_hi);
    double d0 = right - left,
           d1 = top - bottom;
    double x0 = (p.c0 - left) / d0,
           x1 = (p.c1 - bottom) / d1;

    // Hermite 3rd order spline polynomials (in Horner form)
    double hl = ( 2.*x0 - 3.) * x0*x0 + 1.,
           hr = (-2.*x0 + 3.) * x0*x0,
           gl = ((x0 - 2.) * x0 + 1.) * x0 * d0,
           gr = (x0 - 1.) * x0 * x0 * d0,
           hb = ( 2.*x1 - 3.) * x1*x1 + 1.,
           ht = (-2.*x1 + 3.) * x1*x1,
           gb = ((x1 - 2.) * x1 + 1.) * x1 * d1,
           gt = (x1 - 1.) * x1 * x1 * d1;

    std::size_t ilb = this->src_mesh->index(i0_lo, i1_lo),
                ilt = this->src_mesh->index(i0_lo, i1_hi),
                irb = this->src_mesh->index(i0_hi, i1_lo),
                irt = this->src_mesh->index(i0_hi, i1_hi);
    assert(ilb != NOT_INCLUDED);
    assert(ilt != NOT_INCLUDED);
    assert(irb != NOT_INCLUDED);
    assert(irt != NOT_INCLUDED);

    SrcT diff0lb = diff0[ilb],
         diff0lt = diff0[ilt],
         diff0rb = diff0[irb],
         diff0rt = diff0[irt],
         diff1lb = diff1[ilb],
         diff1lt = diff1[ilt],
         diff1rb = diff1[irb],
         diff1rt = diff1[irt];

    SrcT data_lb = this->src_vec[ilb],
         data_lt = this->src_vec[ilt],
         data_rb = this->src_vec[irb],
         data_rt = this->src_vec[irt],
         diff_l = gb * diff1lb + gt * diff1lt,
         diff_r = gb * diff1rb + gt * diff1rt,
         diff_b = gl * diff0lb + gr * diff0rb,
         diff_t = gl * diff0lt + gr * diff0rt;

    return this->flags.postprocess(this->dst_mesh->at(index),
        hl * (hb * data_lb + ht * data_lt) + hr * (hb * data_rb + ht * data_rt) +
        hb * diff_b + ht * diff_t + hl * diff_l + hr * diff_r
    );
}


template <typename DstT, typename SrcT>
SplineFilteredRect3DLazyDataImpl<DstT, SrcT>::SplineFilteredRect3DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh,
                                                                               const DataVector<const SrcT>& src_vec,
                                                                               const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                               const InterpolationFlags& flags):
    InterpolatedLazyDataImpl<DstT, RectangularFilteredMesh3D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
    diff0(src_mesh->size()), diff1(src_mesh->size()), diff2(src_mesh->size()) {}

template <typename DstT, typename SrcT>
DstT SplineFilteredRect3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<3> p;
    size_t i0_lo, i0_hi, i1_lo, i1_hi, i2_lo, i2_hi;

    if (!this->src_mesh->prepareInterpolation(this->dst_mesh->at(index), p, i0_lo, i0_hi, i1_lo, i1_hi, i2_lo, i2_hi, this->flags))
        return NaNfor<decltype(this->src_vec[0])>();

    double back = this->src_mesh->fullMesh.getAxis0()->at(i0_lo), front = this->src_mesh->fullMesh.getAxis0()->at(i0_hi),
           left = this->src_mesh->fullMesh.getAxis1()->at(i1_lo), right = this->src_mesh->fullMesh.getAxis1()->at(i1_hi),
           bottom = this->src_mesh->fullMesh.getAxis2()->at(i2_lo), top = this->src_mesh->fullMesh.getAxis2()->at(i2_hi);

    std::size_t illl = this->src_mesh->index(i0_lo, i1_lo, i2_lo),
                illh = this->src_mesh->index(i0_lo, i1_lo, i2_hi),
                ilhl = this->src_mesh->index(i0_lo, i1_hi, i2_lo),
                ilhh = this->src_mesh->index(i0_lo, i1_hi, i2_hi),
                ihll = this->src_mesh->index(i0_hi, i1_lo, i2_lo),
                ihlh = this->src_mesh->index(i0_hi, i1_lo, i2_hi),
                ihhl = this->src_mesh->index(i0_hi, i1_hi, i2_lo),
                ihhh = this->src_mesh->index(i0_hi, i1_hi, i2_hi);
    assert(illl != NOT_INCLUDED);
    assert(illh != NOT_INCLUDED);
    assert(ilhl != NOT_INCLUDED);
    assert(ilhh != NOT_INCLUDED);
    assert(ihll != NOT_INCLUDED);
    assert(ihlh != NOT_INCLUDED);
    assert(ihhl != NOT_INCLUDED);
    assert(ihhh != NOT_INCLUDED);

    double d0 = front - back,
           d1 = right - left,
           d2 = top - bottom;
    double x0 = (p.c0 - back) / d0,
           x1 = (p.c1 - left) / d1,
           x2 = (p.c2 - bottom) / d2;

    // Hermite 3rd order spline polynomials (in Horner form)
    double h0l = ( 2.*x0 - 3.) * x0*x0 + 1.,
           h0h = (-2.*x0 + 3.) * x0*x0,
           g0l = ((x0 - 2.) * x0 + 1.) * x0 * d0,
           g0h = (x0 - 1.) * x0 * x0 * d0,
           h1l = ( 2.*x1 - 3.) * x1*x1 + 1.,
           h1h = (-2.*x1 + 3.) * x1*x1,
           g1l = ((x1 - 2.) * x1 + 1.) * x1 * d1,
           g1h = (x1 - 1.) * x1 * x1 * d1,
           h2l = ( 2.*x2 - 3.) * x2*x2 + 1.,
           h2h = (-2.*x2 + 3.) * x2*x2,
           g2l = ((x2 - 2.) * x2 + 1.) * x2 * d2,
           g2h = (x2 - 1.) * x2 * x2 * d2;

    SrcT diff0lll = diff0[illl],
         diff0llh = diff0[illh],
         diff0lhl = diff0[ilhl],
         diff0lhh = diff0[ilhh],
         diff0hll = diff0[ihll],
         diff0hlh = diff0[ihlh],
         diff0hhl = diff0[ihhl],
         diff0hhh = diff0[ihhh],
         diff1lll = diff1[illl],
         diff1llh = diff1[illh],
         diff1lhl = diff1[ilhl],
         diff1lhh = diff1[ilhh],
         diff1hll = diff1[ihll],
         diff1hlh = diff1[ihlh],
         diff1hhl = diff1[ihhl],
         diff1hhh = diff1[ihhh],
         diff2lll = diff2[illl],
         diff2llh = diff2[illh],
         diff2lhl = diff2[ilhl],
         diff2lhh = diff2[ilhh],
         diff2hll = diff2[ihll],
         diff2hlh = diff2[ihlh],
         diff2hhl = diff2[ihhl],
         diff2hhh = diff2[ihhh];

    SrcT data_lll = this->src_vec[illl],
         data_llh = this->src_vec[illh],
         data_lhl = this->src_vec[ilhl],
         data_lhh = this->src_vec[ilhh],
         data_hll = this->src_vec[ihll],
         data_hlh = this->src_vec[ihlh],
         data_hhl = this->src_vec[ihhl],
         data_hhh = this->src_vec[ihhh],
         D_ll = g0l * diff0lll + g0h * diff0hll, Dl_l = g1l * diff1lll + g1h * diff1lhl, Dll_ = g2l * diff2lll + g2h * diff2llh,
         D_lh = g0l * diff0llh + g0h * diff0hlh, Dl_h = g1l * diff1llh + g1h * diff1lhh, Dlh_ = g2l * diff2lhl + g2h * diff2lhh,
         D_hl = g0l * diff0lhl + g0h * diff0hhl, Dh_l = g1l * diff1hll + g1h * diff1hhl, Dhl_ = g2l * diff2hll + g2h * diff2hlh,
         D_hh = g0l * diff0lhh + g0h * diff0hhh, Dh_h = g1l * diff1hlh + g1h * diff1hhh, Dhh_ = g2l * diff2hhl + g2h * diff2hhh;

    return this->flags.postprocess(this->dst_mesh->at(index),
        h0l * h1l * h2l * data_lll +
        h0l * h1l * h2h * data_llh +
        h0l * h1h * h2l * data_lhl +
        h0l * h1h * h2h * data_lhh +
        h0h * h1l * h2l * data_hll +
        h0h * h1l * h2h * data_hlh +
        h0h * h1h * h2l * data_hhl +
        h0h * h1h * h2h * data_hhh +
        h1l * h2l * D_ll + h0l * h2l * Dl_l + h0l * h1l * Dll_ +
        h1l * h2h * D_lh + h0l * h2h * Dl_h + h0l * h1h * Dlh_ +
        h1h * h2l * D_hl + h0h * h2l * Dh_l + h0h * h1l * Dhl_ +
        h1h * h2h * D_hh + h0h * h2h * Dh_h + h0h * h1h * Dhh_
    );
}



namespace filtered_hyman {
    template <typename DataT, typename IndexF>
    static void computeDiffs(DataT* diffs, int ax, const shared_ptr<MeshAxis>& axis,
                             const DataT* data, IndexF&& index, const InterpolationFlags& flags)
    {
        const std::size_t n1 = axis->size() - 1;

        for (std::size_t i = 1; i != n1; ++i) {
            const std::size_t idx = index(i);
            if (idx == NOT_INCLUDED) continue;
            const double da = axis->at(i) - axis->at(i-1),
                         db = axis->at(i+1) - axis->at(i);
            size_t idxm = index(i-1), idxp = index(i+1);
            if (idxm == NOT_INCLUDED || idxp == NOT_INCLUDED) {   // at edges derivative is 0
                diffs[idx] = 0. * DataT();
            } else {
                const DataT sa = (data[idx] - data[idxm]) / da,
                            sb = (data[idxp] - data[idx]) / db;
                // Use parabolic estimation of the derivative
                diffs[idx] = (da * sb  + db * sa) / (da + db);
                // Hyman filter
                Hyman<DataT>::filter(diffs[idx], sa, sb);
            }
        }

        const size_t i0 = index(0), i1 = index(1), in = index(n1), in1 = index(n1-1);
        double da0, db0, dan, dbn;
        DataT sa0, sb0, san, sbn;

        if (i0 != NOT_INCLUDED) {
            if (flags.symmetric(ax) && i1 != NOT_INCLUDED) {
                da0 = axis->at(0);
                db0 = axis->at(1) - axis->at(0);
                sb0 = (data[i1] - data[i0]) / db0;
                if (da0 < 0. && flags.periodic(ax)) {
                    da0 += flags.high(ax) - flags.low(ax);
                }
                if (da0 == 0.)
                    sa0 = (data[i1] - flags.reflect(ax, data[i1])) / (2.*db0);
                else if (da0 > 0.)
                    sa0 = (data[i0] - flags.reflect(ax, data[i0])) / (2.*da0);
                else {
                    da0 = db0 = 0.5;
                    sa0 = sb0 = 0. * DataT();
                }
            } else {
                da0 = db0 = 0.5;
                sa0 = sb0 = 0. * DataT();
            }

            // Use parabolic estimation of the derivative with Hyman filter
            diffs[i0] = (da0 * sb0 + db0 * sa0) / (da0 + db0);
            Hyman<DataT>::filter(diffs[i0], sa0, sb0);
        }

        if (in != NOT_INCLUDED) {
            if (flags.symmetric(ax) && in1 != NOT_INCLUDED) {
                dan = axis->at(n1) - axis->at(n1-1);
                san = (data[in] - data[in1]) / dan;
                dbn = - axis->at(n1);
                if (dbn < 0. && flags.periodic(ax)) {
                    dbn += flags.high(ax) - flags.low(ax);
                }
                if (dbn == 0.)
                    sbn = (data[in1] - flags.reflect(ax, data[in1])) / (2.*dan);
                else if (dbn > 0.)
                    sbn = (data[in] - flags.reflect(ax, data[in])) / (2.*dbn);
                else {
                    dan = dbn = 0.5;
                    san = sbn = 0. * DataT();
                }
            } else {
                dan = dbn = 0.5;
                san = sbn = 0. * DataT();
            }

            // Use parabolic estimation of the derivative with Hyman filter
            diffs[in] = (dan * sbn + dbn * san) / (dan + dbn);
            Hyman<DataT>::filter(diffs[in], san, sbn);
        }
    }
}


template <typename DstT, typename SrcT>
HymanSplineFilteredRect2DLazyDataImpl<DstT, SrcT>::HymanSplineFilteredRect2DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh,
                                                                                         const DataVector<const SrcT>& src_vec,
                                                                                         const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                                         const InterpolationFlags& flags):
    SplineFilteredRect2DLazyDataImpl<DstT, SrcT>(src_mesh, src_vec, dst_mesh, flags)
{
    const size_t n0 = src_mesh->fullMesh.axis[0]->size(), n1 = src_mesh->fullMesh.axis[1]->size();

    if (n0 == 0 || n1 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    if (n0 > 1)
        for (size_t i1 = 0; i1 < n1; ++i1)
            filtered_hyman::computeDiffs<SrcT>(this->diff0.data(), 0, src_mesh->fullMesh.axis[0], src_vec.data(),
                                               [&src_mesh, i1](size_t i0) -> size_t { return src_mesh->index(i0, i1); },
                                               flags);
    else
        std::fill(this->diff0.begin(), this->diff0.end(), 0. * SrcT());
    if (n1 > 1)
        for (size_t i0 = 0; i0 < n0; ++i0)
            filtered_hyman::computeDiffs<SrcT>(this->diff1.data(), 1, src_mesh->fullMesh.axis[1], src_vec.data(),
                                               [&src_mesh, i0](size_t i1) -> size_t { return src_mesh->index(i0, i1); },
                                               flags);
    else
        std::fill(this->diff1.begin(), this->diff1.end(), 0. * SrcT());
}



template <typename DstT, typename SrcT>
HymanSplineFilteredRect3DLazyDataImpl<DstT, SrcT>::HymanSplineFilteredRect3DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh,
                                                                                         const DataVector<const SrcT>& src_vec,
                                                                                         const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                                         const InterpolationFlags& flags):
    SplineFilteredRect3DLazyDataImpl<DstT, SrcT>(src_mesh, src_vec, dst_mesh, flags)
{
    const size_t n0 = src_mesh->fullMesh.axis[0]->size(), n1 = src_mesh->fullMesh.axis[1]->size(), n2 = src_mesh->fullMesh.axis[2]->size();

    if (n0 == 0 || n1 == 0 || n2 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    if (n0 > 1) {
        for (size_t i2 = 0; i2 < n2; ++i2) {
            for (size_t i1 = 0; i1 < n1; ++i1) {
                filtered_hyman::computeDiffs<SrcT>(this->diff0.data(), 0, src_mesh->fullMesh.axis[0], src_vec.data(),
                                               [&src_mesh, i2, i1](size_t i0) -> size_t { return src_mesh->index(i0, i1, i2); },
                                               flags);
            }
        }
    } else
        std::fill(this->diff0.begin(), this->diff0.end(), 0. * SrcT());

    if (n1 > 1) {
        for (size_t i2 = 0; i2 < n2; ++i2) {
            for (size_t i0 = 0; i0 < n0; ++i0) {
                filtered_hyman::computeDiffs<SrcT>(this->diff1.data(), 1, src_mesh->fullMesh.axis[1], src_vec.data(),
                                               [&src_mesh, i2, i0](size_t i1) -> size_t { return src_mesh->index(i0, i1, i2); },
                                               flags);
            }
        }
    } else
        std::fill(this->diff1.begin(), this->diff1.end(), 0. * SrcT());

    if (n2 > 1) {
        for (size_t i1 = 0; i1 < n1; ++i1) {
            for (size_t i0 = 0; i0 < n0; ++i0) {
                filtered_hyman::computeDiffs<SrcT>(this->diff2.data(), 2, src_mesh->fullMesh.axis[2], src_vec.data(),
                                               [&src_mesh, i1, i0](size_t i2) -> size_t { return src_mesh->index(i0, i1, i2); },
                                               flags);
            }
        }
    } else
        std::fill(this->diff2.begin(), this->diff2.end(), 0. * SrcT());

}


template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<double, double>;
template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<double, double>;
template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;




} // namespace plask
