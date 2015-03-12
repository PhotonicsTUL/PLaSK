#include "rectangular_spline.h"

namespace plask {

namespace detail {
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
    void computeDiffs(DataT* diffs, int ax, const shared_ptr<RectangularAxis>& axis,
                      const DataT* data, size_t stride, const InterpolationFlags& flags)
    {
        const size_t n0 = axis->size() - 1;

        if (!n0) {
            diffs[0] = 0. * DataT();
            return;
        }

        for (size_t i = 1; i != n0; ++i) {
            const int idx = stride * i;
            const double da = axis->at(i) - axis->at(i-1),
                         db = axis->at(i+1) - axis->at(i);
            const DataT sa = (data[idx] - data[idx-stride]) / da,
                        sb = (data[idx+stride] - data[idx]) / db;
            // Use parabolic estimation of the derivative
            diffs[idx] = (da * sb  + db * sa) / (da + db);
            // Hyman filter
            Hyman<DataT>::filter(diffs[idx], sa, sb);
        }

        const size_t in0 = stride * n0;
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
            dan = axis->at(n0) - axis->at(n0-1);
            san = (data[in0] - data[in0-stride]) / dan;
            dbn = - axis->at(n0);
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
        } else {
            if (flags.periodic(ax)) {
                da0 = axis->at(0) - axis->at(n0) + flags.high(ax) - flags.low(ax);
                db0 = axis->at(1) - axis->at(0);
                sa0 = (data[0] - data[stride*n0]) / da0,
                sb0 = (data[1] - data[0]) / db0;
                dan = axis->at(n0) - axis->at(n0-1);
                dbn = da0;
                san = (data[in0] - data[in0-stride]) / dan,
                sbn = (data[0] - data[in0]) / dbn;
            } else {
                da0 = db0 = dan = dbn = 0.5;
                sa0 = sb0 = san = sbn = 0. * DataT();
            }
        }

        // Use parabolic estimation of the derivative
        diffs[0] = (da0 * sb0  + db0 * sa0) / (da0 + db0);
        diffs[in0] = (dan * sbn  + dbn * san) / (dan + dbn);
        // Hyman filter
        Hyman<DataT>::filter(diffs[0], sa0, sb0);
        Hyman<DataT>::filter(diffs[in0], san, sbn);
    }
}


template <typename DstT, typename SrcT>
SplineRect2DLazyDataImpl<DstT, SrcT>::SplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                                                               const DataVector<const SrcT>& src_vec,
                                                               const shared_ptr<const MeshD<2>>& dst_mesh,
                                                               const InterpolationFlags& flags):
    InterpolatedLazyDataImpl<DstT, RectangularMesh<2>, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
    diff0(src_mesh->size()), diff1(src_mesh->size())
{
    const int n0 = src_mesh->axis0->size(), n1 = src_mesh->axis1->size();

    if (n0 == 0 || n1 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    size_t stride0 = src_mesh->index(1, 0),
           stride1 = src_mesh->index(0, 1);

    if (n0 > 1)
        for (size_t i1 = 0, i = 0; i1 < src_mesh->axis1->size(); ++i1, i += stride1)
            detail::computeDiffs(diff0.data()+i, 0, src_mesh->axis0, src_vec.data()+i, stride0, flags);
    else
        std::fill(diff0.begin(), diff0.end(), 0. * SrcT());
    if (n1 > 1)
        for (size_t i0 = 0, i = 0; i0 < src_mesh->axis0->size(); ++i0, i += stride0)
            detail::computeDiffs(diff1.data()+i, 1, src_mesh->axis1, src_vec.data()+i, stride1, flags);
    else
        std::fill(diff1.begin(), diff1.end(), 0. * SrcT());
}

template <typename DstT, typename SrcT>
DstT SplineRect2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<2> p = this->flags.wrap(this->dst_mesh->at(index));

    size_t i0 = this->src_mesh->axis0->findIndex(p.c0);
    size_t i1 = this->src_mesh->axis1->findIndex(p.c1);

    size_t i0_1;
    double left, right;
    bool invert_left = false, invert_right = false;
    if (i0 == 0) {
        if (this->flags.symmetric(0)) {
            i0_1 = 0;
            left = this->src_mesh->axis0->at(0);
            if (left > 0.) {
                left = - left;
                invert_left = true;
            } else if (this->flags.periodic(0)) {
                left = 2. * this->flags.low(0) - left;
                invert_left = true;
            } else {
                left -= 1.;
            }
        } else if (this->flags.periodic(0)) {
            i0_1 = this->src_mesh->axis0->size() - 1;
            left = this->src_mesh->axis0->at(i0_1) - this->flags.high(0) + this->flags.low(0);
        } else {
            i0_1 = 0;
            left = this->src_mesh->axis0->at(0) - 1.;
        }
    } else {
        i0_1 = i0 - 1;
        left = this->src_mesh->axis0->at(i0_1);
    }
    if (i0 == this->src_mesh->axis0->size()) {
        if (this->flags.symmetric(0)) {
            --i0;
            right = this->src_mesh->axis0->at(i0);
            if (right < 0.) {
                right = - right;
                invert_right = true;
            } else if (this->flags.periodic(0)) {
                left = 2. * this->flags.high(0) - right;
                invert_right = true;
            } else {
                right += 1.;
            }
        } else if (this->flags.periodic(0)) {
            i0 = 0;
            right = this->src_mesh->axis0->at(0) + this->flags.high(0) - this->flags.low(0);
        } else {
            --i0;
            right = this->src_mesh->axis0->at(i0) + 1.;
        }
    } else {
        right = this->src_mesh->axis0->at(i0);
    }

    size_t i1_1;
    double bottom, top;
    bool invert_top = false, invert_bottom = false;
    if (i1 == 0) {
        if (this->flags.symmetric(1)) {
            i1_1 = 0;
            bottom = this->src_mesh->axis1->at(0);
            if (bottom > 0.) {
                bottom = - bottom;
                invert_bottom = true;
            } else if (this->flags.periodic(1)) {
                bottom = 2. * this->flags.low(1) - bottom;
                invert_bottom = true;
            } else {
                bottom -= 1.;
            }
        } else if (this->flags.periodic(1)) {
            i1_1 = this->src_mesh->axis1->size() - 1;
            bottom = this->src_mesh->axis1->at(i1_1) - this->flags.high(1) + this->flags.low(1);
        } else {
            i1_1 = 0;
            bottom = this->src_mesh->axis1->at(0) - 1.;
        }
    } else {
        i1_1 = i1 - 1;
        bottom = this->src_mesh->axis1->at(i1_1);
    }
    if (i1 == this->src_mesh->axis1->size()) {
        if (this->flags.symmetric(1)) {
            --i1;
            top = this->src_mesh->axis1->at(i1);
            if (top < 0.) {
                top = - top;
                invert_top = true;
            } else if (this->flags.periodic(1)) {
                top = 2. * this->flags.high(1) - top;
                invert_top = true;
            } else {
                top += 1.;
            }
        } else if (this->flags.periodic(1)) {
            i1 = 0;
            top = this->src_mesh->axis1->at(0) + this->flags.high(1) - this->flags.low(1);
        } else {
            --i1;
            top = this->src_mesh->axis1->at(i1) + 1.;
        }
    } else {
        top = this->src_mesh->axis1->at(i1);
    }

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

    int ilb = this->src_mesh->index(i0_1, i1_1),
        ilt = this->src_mesh->index(i0_1, i1),
        irb = this->src_mesh->index(i0, i1_1),
        irt = this->src_mesh->index(i0, i1);

    SrcT data_lb = this->src_vec[ilb],
         data_lt = this->src_vec[ilt],
         data_rb = this->src_vec[irb],
         data_rt = this->src_vec[irt],
         diff_l = gb * diff1[ilb] + gt * diff1[ilt],
         diff_r = gb * diff1[irb] + gt * diff1[irt],
         diff_b = gl * diff0[ilb] + gr * diff0[irb],
         diff_t = gl * diff0[ilt] + gr * diff0[irt];

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
SplineRect3DLazyDataImpl<DstT, SrcT>::SplineRect3DLazyDataImpl(const shared_ptr<const RectangularMesh<3>>& src_mesh,
                                                               const DataVector<const SrcT>& src_vec,
                                                               const shared_ptr<const MeshD<3>>& dst_mesh,
                                                               const InterpolationFlags& flags):
    InterpolatedLazyDataImpl<DstT, RectangularMesh<3>, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
    diff0(src_mesh->size()), diff1(src_mesh->size()), diff2(src_mesh->size())
{
    const int n0 = src_mesh->axis0->size(), n1 = src_mesh->axis1->size(), n2 = src_mesh->axis2->size();

    if (n0 == 0 || n1 == 0 || n2 == 0)
        throw BadMesh("interpolate", "Source mesh empty");

    if (n0 > 1) {
        size_t stride0 = src_mesh->index(1, 0, 0);
        for (size_t i2 = 0; i2 < src_mesh->axis2->size(); ++i2) {
            for (size_t i1 = 0; i1 < src_mesh->axis1->size(); ++i1) {
                size_t offset = src_mesh->index(0, i1, i2);
                detail::computeDiffs(diff0.data()+offset, 0, src_mesh->axis0, src_vec.data()+offset, stride0, flags);
            }
        }
    } else
        std::fill(diff0.begin(), diff0.end(), 0. * SrcT());

    if (n1 > 1) {
        size_t stride1 = src_mesh->index(0, 1, 0);
        for (size_t i2 = 0; i2 < src_mesh->axis2->size(); ++i2) {
            for (size_t i0 = 0; i0 < src_mesh->axis0->size(); ++i0) {
                size_t offset = src_mesh->index(i0, 0, i2);
                detail::computeDiffs(diff1.data()+offset, 1, src_mesh->axis1, src_vec.data()+offset, stride1, flags);
            }
        }
    } else
        std::fill(diff1.begin(), diff1.end(), 0. * SrcT());

    if (n2 > 1) {
        size_t stride2 = src_mesh->index(0, 0, 1);
        for (size_t i1 = 0; i1 < src_mesh->axis1->size(); ++i1) {
            for (size_t i0 = 0; i0 < src_mesh->axis0->size(); ++i0) {
                size_t offset = src_mesh->index(i0, i1, 0);
                detail::computeDiffs(diff2.data()+offset, 2, src_mesh->axis2, src_vec.data()+offset, stride2, flags);
            }
        }
    } else
        std::fill(diff2.begin(), diff2.end(), 0. * SrcT());

}

template <typename DstT, typename SrcT>
DstT SplineRect3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<3> p = this->flags.wrap(this->dst_mesh->at(index));

    int i0 = this->src_mesh->axis0->findIndex(p.c0),
        i1 = this->src_mesh->axis1->findIndex(p.c1),
        i2 = this->src_mesh->axis2->findIndex(p.c2);

    size_t i0_1;
    double back, front;
    bool invert_back = false, invert_front = false;
    if (i0 == 0) {
        if (this->flags.symmetric(0)) {
            i0_1 = 0;
            back = this->src_mesh->axis0->at(0);
            if (back > 0.) {
                back = - back;
                invert_back = true;
            } else if (this->flags.periodic(0)) {
                back = 2. * this->flags.low(0) - back;
                invert_back = true;
            } else {
                back -= 1.;
            }
        } else if (this->flags.periodic(0)) {
            i0_1 = this->src_mesh->axis0->size() - 1;
            back = this->src_mesh->axis0->at(i0_1) - this->flags.high(0) + this->flags.low(0);
        } else {
            i0_1 = 0;
            back = this->src_mesh->axis0->at(0) - 1.;
        }
    } else {
        i0_1 = i0 - 1;
        back = this->src_mesh->axis0->at(i0_1);
    }
    if (i0 == this->src_mesh->axis0->size()) {
        if (this->flags.symmetric(0)) {
            --i0;
            front = this->src_mesh->axis0->at(i0);
            if (front < 0.) {
                front = - front;
                invert_front = true;
            } else if (this->flags.periodic(0)) {
                back = 2. * this->flags.high(0) - front;
                invert_front = true;
            } else {
                front += 1.;
            }
        } else if (this->flags.periodic(0)) {
            i0 = 0;
            front = this->src_mesh->axis0->at(0) + this->flags.high(0) - this->flags.low(0);
        } else {
            --i0;
            front = this->src_mesh->axis0->at(i0) + 1.;
        }
    } else {
        front = this->src_mesh->axis0->at(i0);
    }

    size_t i1_1;
    double left, right;
    bool invert_left = false, invert_right = false;
    if (i1 == 0) {
        if (this->flags.symmetric(1)) {
            i1_1 = 0;
            left = this->src_mesh->axis1->at(0);
            if (left > 0.) {
                left = - left;
                invert_left = true;
            } else if (this->flags.periodic(1)) {
                left = 2. * this->flags.low(1) - left;
                invert_left = true;
            } else {
                left -= 1.;
            }
        } else if (this->flags.periodic(1)) {
            i1_1 = this->src_mesh->axis1->size() - 1;
            left = this->src_mesh->axis1->at(i1_1) - this->flags.high(1) + this->flags.low(1);
        } else {
            i1_1 = 0;
            left = this->src_mesh->axis1->at(0) - 1.;
        }
    } else {
        i1_1 = i1 - 1;
        left = this->src_mesh->axis1->at(i1_1);
    }
    if (i1 == this->src_mesh->axis1->size()) {
        if (this->flags.symmetric(1)) {
            --i1;
            right = this->src_mesh->axis1->at(i1);
            if (right < 0.) {
                right = - right;
                invert_right = true;
            } else if (this->flags.periodic(1)) {
                left = 2. * this->flags.high(1) - right;
                invert_right = true;
            } else {
                right += 1.;
            }
        } else if (this->flags.periodic(1)) {
            i1 = 0;
            right = this->src_mesh->axis1->at(0) + this->flags.high(1) - this->flags.low(1);
        } else {
            --i1;
            right = this->src_mesh->axis1->at(i1) + 1.;
        }
    } else {
        right = this->src_mesh->axis1->at(i1);
    }

    size_t i2_1;
    double bottom, top;
    bool invert_top = false, invert_bottom = false;
    if (i2 == 0) {
        if (this->flags.symmetric(2)) {
            i2_1 = 0;
            bottom = this->src_mesh->axis2->at(0);
            if (bottom > 0.) {
                bottom = - bottom;
                invert_bottom = true;
            } else if (this->flags.periodic(2)) {
                bottom = 2. * this->flags.low(2) - bottom;
                invert_bottom = true;
            } else {
                bottom -= 1.;
            }
        } else if (this->flags.periodic(2)) {
            i2_1 = this->src_mesh->axis2->size() - 1;
            bottom = this->src_mesh->axis2->at(i2_1) - this->flags.high(2) + this->flags.low(2);
        } else {
            i2_1 = 0;
            bottom = this->src_mesh->axis2->at(0) - 1.;
        }
    } else {
        i2_1 = i2-1;
        bottom = this->src_mesh->axis2->at(i2_1);
    }
    if (i2 == this->src_mesh->axis2->size()) {
        if (this->flags.symmetric(2)) {
            --i2;
            top = this->src_mesh->axis2->at(i2);
            if (top < 0.) {
                top = - top;
                invert_top = true;
            } else if (this->flags.periodic(2)) {
                top = 2. * this->flags.high(2) - top;
                invert_top = true;
            } else {
                top += 1.;
            }
        } else if (this->flags.periodic(2)) {
            i2 = 0;
            top = this->src_mesh->axis2->at(0) + this->flags.high(2) - this->flags.low(2);
        } else {
            --i2;
            top = this->src_mesh->axis2->at(i2) + 1.;
        }
    } else {
        top = this->src_mesh->axis2->at(i2);
    }

    int illl = this->src_mesh->index(i0_1, i1_1, i2_1),
        illh = this->src_mesh->index(i0_1, i1_1, i2),
        ilhl = this->src_mesh->index(i0_1, i1, i2_1),
        ilhh = this->src_mesh->index(i0_1, i1, i2),
        ihll = this->src_mesh->index(i0, i1_1, i2_1),
        ihlh = this->src_mesh->index(i0, i1_1, i2),
        ihhl = this->src_mesh->index(i0, i1, i2_1),
        ihhh = this->src_mesh->index(i0, i1, i2);

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

    SrcT data_lll = this->src_vec[illl],
         data_llh = this->src_vec[illh],
         data_lhl = this->src_vec[ilhl],
         data_lhh = this->src_vec[ilhh],
         data_hll = this->src_vec[ihll],
         data_hlh = this->src_vec[ihlh],
         data_hhl = this->src_vec[ihhl],
         data_hhh = this->src_vec[ihhh],
         D_ll = g0l * diff0[illl] + g0h * diff0[ihll], Dl_l = g1l * diff1[illl] + g1h * diff1[ilhl], Dll_ = g2l * diff2[illl] + g2h * diff2[illh],
         D_lh = g0l * diff0[illh] + g0h * diff0[ihlh], Dl_h = g1l * diff1[illh] + g1h * diff1[ilhh], Dlh_ = g2l * diff2[ilhl] + g2h * diff2[ilhh],
         D_hl = g0l * diff0[ilhl] + g0h * diff0[ihhl], Dh_l = g1l * diff1[ihll] + g1h * diff1[ihhl], Dhl_ = g2l * diff2[ihll] + g2h * diff2[ihlh],
         D_hh = g0l * diff0[ilhh] + g0h * diff0[ihhh], Dh_h = g1l * diff1[ihlh] + g1h * diff1[ihhh], Dhh_ = g2l * diff2[ihhl] + g2h * diff2[ihhh];


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


template struct PLASK_API SplineRect2DLazyDataImpl<double, double>;
template struct PLASK_API SplineRect2DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API SplineRect2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API SplineRect2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API SplineRect2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API SplineRect2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API SplineRect2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API SplineRect2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API SplineRect2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API SplineRect2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;

template struct PLASK_API SplineRect3DLazyDataImpl<double, double>;
template struct PLASK_API SplineRect3DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API SplineRect3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API SplineRect3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API SplineRect3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API SplineRect3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API SplineRect3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API SplineRect3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API SplineRect3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API SplineRect3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;

} // namespace plask
