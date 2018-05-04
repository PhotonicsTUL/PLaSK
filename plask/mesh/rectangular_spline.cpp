#include "rectangular_spline.h"
#include "../exceptions.h"


namespace plask {


template <typename DstT, typename SrcT>
SplineRect2DLazyDataImpl<DstT, SrcT>::SplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh2D>& src_mesh,
                                                                       const DataVector<const SrcT>& src_vec,
                                                                       const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                       const InterpolationFlags& flags):
    InterpolatedLazyDataImpl<DstT, RectangularMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
    diff0(src_mesh->size()), diff1(src_mesh->size()) {}

template <typename DstT, typename SrcT>
DstT SplineRect2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<2> p = this->flags.wrap(this->dst_mesh->at(index));

    size_t i0_lo, i0_hi;
    double left, right;
    bool invert_left, invert_right;
    prepareInterpolationForAxis(*this->src_mesh->axis[0], this->flags, p.c0, 0, i0_lo, i0_hi, left, right, invert_left, invert_right);

    size_t i1_lo, i1_hi;
    double bottom, top;
    bool invert_bottom, invert_top;
    prepareInterpolationForAxis(*this->src_mesh->axis[1], this->flags, p.c1, 1, i1_lo, i1_hi, bottom, top, invert_bottom, invert_top);

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

    SrcT diff0lb = diff0[ilb],
         diff0lt = diff0[ilt],
         diff0rb = diff0[irb],
         diff0rt = diff0[irt],
         diff1lb = diff1[ilb],
         diff1lt = diff1[ilt],
         diff1rb = diff1[irb],
         diff1rt = diff1[irt];

    if (invert_left)   { diff0lb = -this->flags.reflect(0, diff0lb); diff0lt = -this->flags.reflect(0, diff0lt); };
    if (invert_right)  { diff0rb = -this->flags.reflect(0, diff0rb); diff0rt = -this->flags.reflect(0, diff0rt); };
    if (invert_top)    { diff1lt = -this->flags.reflect(1, diff1lt); diff1rt = -this->flags.reflect(1, diff1rt); };
    if (invert_bottom) { diff1lb = -this->flags.reflect(1, diff1lb); diff1rb = -this->flags.reflect(1, diff1rb); };

    SrcT data_lb = this->src_vec[ilb],
         data_lt = this->src_vec[ilt],
         data_rb = this->src_vec[irb],
         data_rt = this->src_vec[irt],
         diff_l = gb * diff1lb + gt * diff1lt,
         diff_r = gb * diff1rb + gt * diff1rt,
         diff_b = gl * diff0lb + gr * diff0rb,
         diff_t = gl * diff0lt + gr * diff0rt;

    if (invert_left)   { data_lb = this->flags.reflect(0, data_lb); data_lt = this->flags.reflect(0, data_lt); diff_l = this->flags.reflect(0, diff_l); }
    if (invert_right)  { data_rb = this->flags.reflect(0, data_rb); data_rt = this->flags.reflect(0, data_rt); diff_r = this->flags.reflect(0, diff_r); }
    if (invert_top)    { data_lt = this->flags.reflect(1, data_lt); data_rt = this->flags.reflect(1, data_rt); diff_t = this->flags.reflect(1, diff_t); }
    if (invert_bottom) { data_lb = this->flags.reflect(1, data_lb); data_rb = this->flags.reflect(1, data_rb); diff_b = this->flags.reflect(1, diff_b); }

    return this->flags.postprocess(this->dst_mesh->at(index),
        hl * (hb * data_lb + ht * data_lt) + hr * (hb * data_rb + ht * data_rt) +
        hb * diff_b + ht * diff_t + hl * diff_l + hr * diff_r
    );
}


template <typename DstT, typename SrcT>
SplineRect3DLazyDataImpl<DstT, SrcT>::SplineRect3DLazyDataImpl(const shared_ptr<const RectilinearMesh3D>& src_mesh,
                                                                       const DataVector<const SrcT>& src_vec,
                                                                       const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                       const InterpolationFlags& flags):
    InterpolatedLazyDataImpl<DstT, RectilinearMesh3D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
    diff0(src_mesh->size()), diff1(src_mesh->size()), diff2(src_mesh->size()) {}

template <typename DstT, typename SrcT>
DstT SplineRect3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<3> p = this->flags.wrap(this->dst_mesh->at(index));

    size_t i0_lo, i0_hi;
    double back, front;
    bool invert_back, invert_front;
    prepareInterpolationForAxis(*this->src_mesh->axis[0], this->flags, p.c0, 0, i0_lo, i0_hi, back, front, invert_back, invert_front);

    size_t i1_lo, i1_hi;
    double left, right;
    bool invert_left, invert_right;
    prepareInterpolationForAxis(*this->src_mesh->axis[1], this->flags, p.c1, 1, i1_lo, i1_hi, left, right, invert_left, invert_right);

    size_t i2_lo, i2_hi;
    double bottom, top;
    bool invert_bottom, invert_top;
    prepareInterpolationForAxis(*this->src_mesh->axis[2], this->flags, p.c2, 2, i2_lo, i2_hi, bottom, top, invert_bottom, invert_top);

    std::size_t illl = this->src_mesh->index(i0_lo, i1_lo, i2_lo),
                illh = this->src_mesh->index(i0_lo, i1_lo, i2_hi),
                ilhl = this->src_mesh->index(i0_lo, i1_hi, i2_lo),
                ilhh = this->src_mesh->index(i0_lo, i1_hi, i2_hi),
                ihll = this->src_mesh->index(i0_hi, i1_lo, i2_lo),
                ihlh = this->src_mesh->index(i0_hi, i1_lo, i2_hi),
                ihhl = this->src_mesh->index(i0_hi, i1_hi, i2_lo),
                ihhh = this->src_mesh->index(i0_hi, i1_hi, i2_hi);

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

    if (invert_back)   { diff0lll = -this->flags.reflect(0, diff0lll); diff0llh = -this->flags.reflect(0, diff0llh);
                         diff0lhl = -this->flags.reflect(0, diff0lhl); diff0lhh = -this->flags.reflect(0, diff0lhh); };
    if (invert_front)  { diff0hll = -this->flags.reflect(0, diff0hll); diff0hlh = -this->flags.reflect(0, diff0hlh);
                         diff0hhl = -this->flags.reflect(0, diff0hhl); diff0hhh = -this->flags.reflect(0, diff0hhh); };
    if (invert_left)   { diff1lll = -this->flags.reflect(1, diff1lll); diff0llh = -this->flags.reflect(1, diff1llh);
                         diff1hll = -this->flags.reflect(1, diff1hll); diff0hlh = -this->flags.reflect(1, diff1hlh); };
    if (invert_right)  { diff1lhl = -this->flags.reflect(1, diff1lhl); diff0lhh = -this->flags.reflect(1, diff1lhh);
                         diff1hhl = -this->flags.reflect(1, diff1hhl); diff0hhh = -this->flags.reflect(1, diff1hhh); };
    if (invert_top)    { diff2lll = -this->flags.reflect(2, diff2lll); diff0lhl = -this->flags.reflect(2, diff2lhl);
                         diff2hll = -this->flags.reflect(2, diff2hll); diff0hhl = -this->flags.reflect(2, diff2hhl); };
    if (invert_bottom) { diff2llh = -this->flags.reflect(2, diff2llh); diff0lhh = -this->flags.reflect(2, diff2lhh);
                         diff2hlh = -this->flags.reflect(2, diff2hlh); diff0hhh = -this->flags.reflect(2, diff2hhh); };


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


    if (invert_back)   { data_lll = this->flags.reflect(0, data_lll); data_llh = this->flags.reflect(0, data_llh);
                         data_lhl = this->flags.reflect(0, data_lhl); data_lhh = this->flags.reflect(0, data_lhh);
                         Dl_l = this->flags.reflect(0, Dl_l); Dl_h = this->flags.reflect(0, Dl_h);
                         Dll_ = this->flags.reflect(0, Dll_); Dlh_ = this->flags.reflect(0, Dlh_); }
    if (invert_front)  { data_hll = this->flags.reflect(0, data_hll); data_llh = this->flags.reflect(0, data_hlh);
                         data_lhl = this->flags.reflect(0, data_hhl); data_lhh = this->flags.reflect(0, data_hhh);
                         Dh_l = this->flags.reflect(0, Dh_l); Dh_h = this->flags.reflect(0, Dh_h);
                         Dhl_ = this->flags.reflect(0, Dhl_); Dhh_ = this->flags.reflect(0, Dhh_); }
    if (invert_left)   { data_lll = this->flags.reflect(1, data_lll); data_llh = this->flags.reflect(1, data_llh);
                         data_hll = this->flags.reflect(1, data_hll); data_hlh = this->flags.reflect(1, data_hlh);
                         Dll_ = this->flags.reflect(1, Dll_); Dhl_ = this->flags.reflect(1, Dhl_);
                         D_ll = this->flags.reflect(1, D_ll); D_lh = this->flags.reflect(1, D_lh); }
    if (invert_right)  { data_lhl = this->flags.reflect(1, data_lhl); data_llh = this->flags.reflect(1, data_lhh);
                         data_hll = this->flags.reflect(1, data_hhl); data_hlh = this->flags.reflect(1, data_hhh);
                         Dlh_ = this->flags.reflect(1, Dlh_); Dhh_ = this->flags.reflect(1, Dhh_);
                         D_hl = this->flags.reflect(1, D_hl); D_hh = this->flags.reflect(1, D_hh); }
    if (invert_bottom) { data_lll = this->flags.reflect(2, data_lll); data_lhl = this->flags.reflect(2, data_lhl);
                         data_hll = this->flags.reflect(2, data_hll); data_hhl = this->flags.reflect(2, data_hhl);
                         D_ll = this->flags.reflect(2, D_ll); D_hl = this->flags.reflect(2, D_hl);
                         Dl_l = this->flags.reflect(2, Dl_l); Dh_l = this->flags.reflect(2, Dh_l); }
    if (invert_top)    { data_llh = this->flags.reflect(2, data_llh); data_lhl = this->flags.reflect(2, data_lhh);
                         data_hll = this->flags.reflect(2, data_hlh); data_hhl = this->flags.reflect(2, data_hhh);
                         D_lh = this->flags.reflect(2, D_lh); D_hh = this->flags.reflect(2, D_hh);
                         Dl_h = this->flags.reflect(2, Dl_h); Dh_h = this->flags.reflect(2, Dh_h); }

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



namespace hyman {
    template <typename T> struct Hyman {
        static void filter(T& data, const T& a, const T& b) {
            T lim = 3 * min(abs(a), abs(b));
            if (data > lim) data = lim;
            else if (data < -lim) data = -lim;
        }
    };

    template <> struct Hyman<dcomplex> {
        static void filter(dcomplex& data, const dcomplex& a, const dcomplex& b) {
            double re = data.real(), im = data.imag();
            Hyman<double>::filter(re, real(a), real(b));
            Hyman<double>::filter(im, imag(a), imag(b));
            data = dcomplex(re,im);
        }
    };

    template <typename T> struct Hyman<Vec<2,T>> {
        static void filter(Vec<2,T>& data, const Vec<2,T>& a, const Vec<2,T>& b) {
            Hyman<T>::filter(data.c0, a.c0, b.c0);
            Hyman<T>::filter(data.c1, a.c1, b.c1);
        }
    };

    template <typename T> struct Hyman<Vec<3,T>> {
        static void filter(Vec<3,T>& data, const Vec<3,T>& a, const Vec<3,T>& b) {
            Hyman<T>::filter(data.c0, a.c0, b.c0);
            Hyman<T>::filter(data.c1, a.c1, b.c1);
            Hyman<T>::filter(data.c2, a.c2, b.c2);
        }
    };

    template <typename T> struct Hyman<Tensor2<T>> {
        static void filter(Tensor2<T>& data, const Tensor2<T>& a, const Tensor2<T>& b) {
            Hyman<T>::filter(data.c00, a.c00, b.c00);
            Hyman<T>::filter(data.c11, a.c11, b.c11);
        }
    };

    template <typename T> struct Hyman<Tensor3<T>> {
        static void filter(Tensor3<T>& data, const Tensor3<T>& a, const Tensor3<T>& b) {
            Hyman<T>::filter(data.c00, a.c00, b.c00);
            Hyman<T>::filter(data.c11, a.c11, b.c11);
            Hyman<T>::filter(data.c22, a.c22, b.c22);
            Hyman<T>::filter(data.c01, a.c01, b.c01);
        }
    };

    template <typename DataT>
    static void computeDiffs(DataT* diffs, int ax, const shared_ptr<MeshAxis>& axis,
                             const DataT* data, std::size_t stride, const InterpolationFlags& flags)
    {
        const std::size_t n1 = axis->size() - 1;

        for (std::size_t i = 1; i != n1; ++i) {
            const std::size_t idx = stride * i;
            const double da = axis->at(i) - axis->at(i-1),
                         db = axis->at(i+1) - axis->at(i);
            const DataT sa = (data[idx] - data[idx-stride]) / da,
                        sb = (data[idx+stride] - data[idx]) / db;
            // Use parabolic estimation of the derivative
            diffs[idx] = (da * sb  + db * sa) / (da + db);
            // Hyman filter
            Hyman<DataT>::filter(diffs[idx], sa, sb);
        }

        const size_t in0 = stride * n1;
        double da0, db0, dan, dbn;
        DataT sa0, sb0, san, sbn;

        if (flags.symmetric(ax)) {
            da0 = axis->at(0);
            db0 = axis->at(1) - axis->at(0);
            sb0 = (data[1] - data[0]) / db0;
            if (da0 < 0. && flags.periodic(ax)) {
                da0 += flags.high(ax) - flags.low(ax);
            }
            if (da0 == 0.)
                sa0 = (data[1] - flags.reflect(ax, data[1])) / (2.*db0);
            else if (da0 > 0.)
                sa0 = (data[0] - flags.reflect(ax, data[0])) / (2.*da0);
            else {
                da0 = db0 = 0.5;
                sa0 = sb0 = 0. * DataT();
            }
            dan = axis->at(n1) - axis->at(n1-1);
            san = (data[in0] - data[in0-stride]) / dan;
            dbn = - axis->at(n1);
            if (dbn < 0. && flags.periodic(ax)) {
                dbn += flags.high(ax) - flags.low(ax);
            }
            if (dbn == 0.)
                sbn = (data[in0-stride] - flags.reflect(ax, data[in0-stride])) / (2.*dan);
            else if (dbn > 0.)
                sbn = (data[in0] - flags.reflect(ax, data[in0])) / (2.*dbn);
            else {
                dan = dbn = 0.5;
                san = sbn = 0. * DataT();
            }
        } else if (flags.periodic(ax)) {
            da0 = axis->at(0) - axis->at(n1) + flags.high(ax) - flags.low(ax);
            db0 = axis->at(1) - axis->at(0);
            dan = axis->at(n1) - axis->at(n1-1);
            dbn = da0;
            sb0 = (data[1] - data[0]) / db0;
            san = (data[in0] - data[in0-stride]) / dan;
            if (da0 == 0.) { da0 = dan; sa0 = san; }
            else sa0 = (data[0] - data[in0]) / da0;
            if (dbn == 0.) {  dbn = db0; sbn = sb0; }
            else sbn = (data[0] - data[in0]) / dbn;
        } else {
            da0 = db0 = dan = dbn = 0.5;
            sa0 = sb0 = san = sbn = 0. * DataT();
        }

        // Use parabolic estimation of the derivative
        diffs[0] = (da0 * sb0 + db0 * sa0) / (da0 + db0);
        diffs[in0] = (dan * sbn + dbn * san) / (dan + dbn);
        // Hyman filter
        Hyman<DataT>::filter(diffs[0], sa0, sb0);
        Hyman<DataT>::filter(diffs[in0], san, sbn);
    }
}


template <typename DstT, typename SrcT>
HymanSplineRect2DLazyDataImpl<DstT, SrcT>::HymanSplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh2D>& src_mesh,
                                                                         const DataVector<const SrcT>& src_vec,
                                                                         const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                         const InterpolationFlags& flags):
    SplineRect2DLazyDataImpl<DstT, SrcT>(src_mesh, src_vec, dst_mesh, flags)
{
    const int n0 = int(src_mesh->axis[0]->size()), n1 = int(src_mesh->axis[1]->size());

    if (n0 == 0 || n1 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    size_t stride0 = src_mesh->index(1, 0),
           stride1 = src_mesh->index(0, 1);

    if (n0 > 1)
        for (size_t i1 = 0, i = 0; i1 < src_mesh->axis[1]->size(); ++i1, i += stride1)
            hyman::computeDiffs<SrcT>(this->diff0.data()+i, 0, src_mesh->axis[0], src_vec.data()+i, stride0, flags);
    else
        std::fill(this->diff0.begin(), this->diff0.end(), 0. * SrcT());
    if (n1 > 1)
        for (size_t i0 = 0, i = 0; i0 < src_mesh->axis[0]->size(); ++i0, i += stride0)
            hyman::computeDiffs<SrcT>(this->diff1.data()+i, 1, src_mesh->axis[1], src_vec.data()+i, stride1, flags);
    else
        std::fill(this->diff1.begin(), this->diff1.end(), 0. * SrcT());
}



template <typename DstT, typename SrcT>
HymanSplineRect3DLazyDataImpl<DstT, SrcT>::HymanSplineRect3DLazyDataImpl(const shared_ptr<const RectilinearMesh3D>& src_mesh,
                                                                         const DataVector<const SrcT>& src_vec,
                                                                         const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                         const InterpolationFlags& flags):
    SplineRect3DLazyDataImpl<DstT, SrcT>(src_mesh, src_vec, dst_mesh, flags)
{
    const int n0 = int(src_mesh->axis[0]->size()), n1 = int(src_mesh->axis[1]->size()), n2 = int(src_mesh->axis[2]->size());

    if (n0 == 0 || n1 == 0 || n2 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    if (n0 > 1) {
        size_t stride0 = src_mesh->index(1, 0, 0);
        for (size_t i2 = 0; i2 < src_mesh->axis[2]->size(); ++i2) {
            for (size_t i1 = 0; i1 < src_mesh->axis[1]->size(); ++i1) {
                size_t offset = src_mesh->index(0, i1, i2);
                hyman::computeDiffs<SrcT>(this->diff0.data()+offset, 0, src_mesh->axis[0], src_vec.data()+offset, stride0, flags);
            }
        }
    } else
        std::fill(this->diff0.begin(), this->diff0.end(), 0. * SrcT());

    if (n1 > 1) {
        size_t stride1 = src_mesh->index(0, 1, 0);
        for (size_t i2 = 0; i2 < src_mesh->axis[2]->size(); ++i2) {
            for (size_t i0 = 0; i0 < src_mesh->axis[0]->size(); ++i0) {
                size_t offset = src_mesh->index(i0, 0, i2);
                hyman::computeDiffs<SrcT>(this->diff1.data()+offset, 1, src_mesh->axis[1], src_vec.data()+offset, stride1, flags);
            }
        }
    } else
        std::fill(this->diff1.begin(), this->diff1.end(), 0. * SrcT());

    if (n2 > 1) {
        size_t stride2 = src_mesh->index(0, 0, 1);
        for (size_t i1 = 0; i1 < src_mesh->axis[1]->size(); ++i1) {
            for (size_t i0 = 0; i0 < src_mesh->axis[0]->size(); ++i0) {
                size_t offset = src_mesh->index(i0, i1, 0);
                hyman::computeDiffs<SrcT>(this->diff2.data()+offset, 2, src_mesh->axis[2], src_vec.data()+offset, stride2, flags);
            }
        }
    } else
        std::fill(this->diff2.begin(), this->diff2.end(), 0. * SrcT());

}


template struct PLASK_API HymanSplineRect2DLazyDataImpl<double, double>;
template struct PLASK_API HymanSplineRect2DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API HymanSplineRect2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API HymanSplineRect2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API HymanSplineRect2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API HymanSplineRect2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API HymanSplineRect2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API HymanSplineRect2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API HymanSplineRect2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API HymanSplineRect2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;

template struct PLASK_API HymanSplineRect3DLazyDataImpl<double, double>;
template struct PLASK_API HymanSplineRect3DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API HymanSplineRect3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API HymanSplineRect3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API HymanSplineRect3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API HymanSplineRect3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API HymanSplineRect3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API HymanSplineRect3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API HymanSplineRect3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API HymanSplineRect3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;



namespace spline {
    template <typename DataT>
    static void computeDiffs(DataT* data, size_t stride, size_t stride1, size_t size1, size_t stride2, size_t size2,
                      int ax, const shared_ptr<MeshAxis>& axis, const InterpolationFlags& flags)
    {
        const size_t n0 = axis->size();
        const size_t n1 = n0 - 1;

		std::unique_ptr<double[]>
			dl(new double[n1]),
			dd(new double[n1 + 1]),
			du(new double[n1]);

        std::unique_ptr<DataT[]> lastdata(new DataT[size1*size2]);
        {
            double left = (axis->at(0) >= 0.)? 0. : flags.low(ax);
            const double da = 2. * (axis->at(0) - left), db = axis->at(1) - axis->at(0);
            const double dab = da/db, dba = db/da;
            for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                size_t cc = c1 * size2;
                for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2) {
                    lastdata[cc+c2] = data[s1+s2];
                    if (da == 0.) {
                         data[s1+s2] = 0.5 * (data[s1+s2+stride] - flags.reflect(ax, data[s1+s2+stride]))  / db;
                    } else {
                        data[s1+s2] = 3. * (dab * data[s1+s2+stride] + (dba-dab) * lastdata[cc+c2] - dba * flags.reflect(ax, lastdata[cc+c2]));
                    }
                }
            }
        }

        for (size_t i = 1, si = stride; i != n1; ++i, si += stride) {
            const double da = axis->at(i) - axis->at(i-1),      // d[i-1]
                         db = axis->at(i+1) - axis->at(i);      // d[i]
            const double dab = da/db, dba = db/da;
            dl[i-1] = db;
            dd[i] = 2. * (da + db);
            du[i] = da;
            for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                size_t cc = c1 * size2;
                for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2) {
                    DataT current = data[s1+s2+si];
                    data[s1+s2+si] = 3. * (dab * data[s1+s2+si+stride] + (dba-dab) * current - dba * lastdata[cc+c2]);
                    lastdata[cc+c2] = current;
                }
            }
        }

        if (!flags.periodic(ax) || flags.symmetric(ax)) {

            if (flags.symmetric(ax)) {
                if (axis->at(0) == 0. || (flags.periodic(ax) && is_zero(axis->at(0) - flags.low(ax)))) {
                    dd[0] = 1.;
                    du[0] = 0.;
                } else if (axis->at(0) > 0. || flags.periodic(ax)) {
                    const double left = (axis->at(0) >= 0.)? 0. : flags.low(ax);
                    // dd[0] = 3.*axis->at(0) + axis->at(1) - 4.*left;
                    dd[0] = 2.*axis->at(0) + 2.*axis->at(1) - 4.*left;  // hack, but should work for asymmetric as well
                    du[0] = 2. * (axis->at(0) - left);
                } else {
                    dd[0] = 1.;
                    du[0] = 0.;
                    for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1)
                        for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                            data[s1+s2] = 0. * DataT();
                }
                if (axis->at(n1) == 0. || (flags.periodic(ax) && is_zero(flags.high(ax) - axis->at(n1)))) {
                    dd[n1] = 1.;
                    dl[n1-1] = 0.;
                    size_t ns = n1 * stride;
                    double ih = 0.5 / (axis->at(n1) - axis->at(n1-1));
                    for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                        size_t cc = c1 * size2;
                        for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2) {
                            data[s1+s2+ns] = (flags.reflect(ax, lastdata[cc+c2]) - lastdata[cc+c2]) * ih;
                        }
                    }
                } else if (axis->at(n1) < 0. || flags.periodic(ax)) {
                    const double right = (axis->at(n1) <= 0.)? 0. : flags.high(ax);
                    const double da = axis->at(n1) - axis->at(n1-1), db = 2. * (right - axis->at(n1));
                    const double dab = da/db, dba = db/da;
                    dl[n1-1] = db;
                    // dd[n1] = 4.*right - 3.*axis->at(n1) - axis->at(n1-1);
                    dd[n1] = 4.*right - 2.*axis->at(n1) - 2.*axis->at(n1-1);  // hack
                    size_t ns = n1 * stride;
                    for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                        size_t cc = c1 * size2;
                        for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                            data[s1+s2+ns] = 3. * (dab * flags.reflect(ax, data[s1+s2+ns]) + (dba-dab) * data[s1+s2+ns] - dba * lastdata[cc+c2]);
                    }
                } else {
                    dl[n1-1] = 0.;
                    dd[n1] = 1.;
                    size_t ns = n1 * stride;
                    for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                        for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                            data[s1+s2+ns] = 0. * DataT();
                    }
                }
            } else {
                du[0] = dl[n1-1] = 0.;
                dd[0] = dd[n1] = 1.;
                size_t ns = n1 * stride;
                for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                    for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                        data[s1+s2] = data[s1+s2+ns] = 0. * DataT();
                }
            }

            // Thomas algorithm
            double id0 = 1. / dd[0];
            du[0] *= id0;
            for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                    data[s1+s2] *= id0;
            }

            /* loop from 1 to X - 1 inclusive, performing the forward sweep */
            for (size_t i = 1, si = stride; i < n0; i++, si += stride) {
                const double m = 1. / (dd[i] - dl[i-1] * du[i-1]);
                du[i] *= m;
                for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                    for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                        data[s1+s2+si] = (data[s1+s2+si] - dl[i-1] * data[s1+s2+si-stride]) * m;
                }
            }

            /* loop from X - 2 to 0 inclusive (safely testing loop condition for an unsigned integer), to perform the back substitution */
            for (size_t i = n1, si = n1*stride; i-- > 0; si -= stride) {
                for (size_t c1 = 0, s1 = 0; c1 != size1; ++c1, s1 += stride1) {
                    for (size_t c2 = 0, s2 = 0; c2 != size2; ++c2, s2 += stride2)
                        data[s1+s2+si-stride] -= du[i] * data[s1+s2+si];
                }
            }
        } else {
            throw NotImplemented("smooth spline for periodic, non-symmetric geometry");
        }
    }

}


template <typename DstT, typename SrcT>
SmoothSplineRect2DLazyDataImpl<DstT, SrcT>::SmoothSplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh2D>& src_mesh,
                                                                           const DataVector<const SrcT>& src_vec,
                                                                           const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                           const InterpolationFlags& flags):
    SplineRect2DLazyDataImpl<DstT, SrcT>(src_mesh, src_vec, dst_mesh, flags)
{
    const size_t n0 = int(src_mesh->axis[0]->size()), n1 = int(src_mesh->axis[1]->size());

    if (n0 == 0 || n1 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    size_t stride0 = src_mesh->index(1, 0),
           stride1 = src_mesh->index(0, 1);

    DataVector<double> data;

    if (n0 > 1) {
        std::copy(src_vec.begin(), src_vec.end(), this->diff0.begin());
        spline::computeDiffs<SrcT>(this->diff0.data(), stride0, stride1, src_mesh->axis[1]->size(), 0, 1, 0, src_mesh->axis[0], flags);
    } else {
        std::fill(this->diff0.begin(), this->diff0.end(), 0. * SrcT());
    }
    if (n1 > 1) {
        std::copy(src_vec.begin(), src_vec.end(), this->diff1.begin());
        spline::computeDiffs<SrcT>(this->diff1.data(), stride1, stride0, src_mesh->axis[0]->size(), 0, 1, 1, src_mesh->axis[1], flags);
    } else {
        std::fill(this->diff1.begin(), this->diff1.end(), 0. * SrcT());
    }
}


template <typename DstT, typename SrcT>
SmoothSplineRect3DLazyDataImpl<DstT, SrcT>::SmoothSplineRect3DLazyDataImpl(const shared_ptr<const RectilinearMesh3D>& src_mesh,
                                                                           const DataVector<const SrcT>& src_vec,
                                                                           const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                           const InterpolationFlags& flags):
    SplineRect3DLazyDataImpl<DstT, SrcT>(src_mesh, src_vec, dst_mesh, flags)
{
    const size_t n0 = int(src_mesh->axis[0]->size()), n1 = int(src_mesh->axis[1]->size()), n2 = int(src_mesh->axis[2]->size());

    if (n0 == 0 || n1 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    size_t stride0 = src_mesh->index(1, 0, 0),
           stride1 = src_mesh->index(0, 1, 0),
           stride2 = src_mesh->index(0, 0, 1);

    DataVector<double> data;

    if (n0 > 1) {
        std::copy(src_vec.begin(), src_vec.end(), this->diff0.begin());
        spline::computeDiffs<SrcT>(this->diff0.data(), stride0,
                                   stride1, src_mesh->axis[1]->size(), stride2, src_mesh->axis[2]->size(),
                                   0, src_mesh->axis[0], flags);
    } else {
        std::fill(this->diff0.begin(), this->diff0.end(), 0. * SrcT());
    }
    if (n1 > 1) {
        std::copy(src_vec.begin(), src_vec.end(), this->diff1.begin());
        spline::computeDiffs<SrcT>(this->diff1.data(), stride1,
                                   stride0, src_mesh->axis[0]->size(), stride2, src_mesh->axis[2]->size(),
                                   1, src_mesh->axis[1], flags);
    } else {
        std::fill(this->diff1.begin(), this->diff1.end(), 0. * SrcT());
    }
    if (n2 > 1) {
        std::copy(src_vec.begin(), src_vec.end(), this->diff2.begin());
        spline::computeDiffs<SrcT>(this->diff2.data(), stride2,
                                   stride0, src_mesh->axis[0]->size(), stride1, src_mesh->axis[1]->size(),
                                   2, src_mesh->axis[2], flags);
    } else {
        std::fill(this->diff2.begin(), this->diff2.end(), 0. * SrcT());
    }
}


template struct PLASK_API SmoothSplineRect2DLazyDataImpl<double, double>;
template struct PLASK_API SmoothSplineRect2DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API SmoothSplineRect2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


template struct PLASK_API SmoothSplineRect3DLazyDataImpl<double, double>;
template struct PLASK_API SmoothSplineRect3DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API SmoothSplineRect3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;



} // namespace plask
