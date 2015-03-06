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
    void computeDiffs(DataT* diffs, const shared_ptr<RectangularAxis>& axis, const DataT* data, size_t stride)
    {
        const size_t n0 = axis->size() - 1;
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
        diffs[0] = diffs[stride*n0] = 0. * DataT();
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
            detail::computeDiffs(diff0.data()+i, src_mesh->axis0, src_vec.data()+i, stride0);
    if (n1 > 1)
        for (size_t i0 = 0, i = 0; i0 < src_mesh->axis0->size(); ++i0, i += stride0)
            detail::computeDiffs(diff1.data()+i, src_mesh->axis1, src_vec.data()+i, stride1);
}

template <typename DstT, typename SrcT>
DstT SplineRect2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    const int n0 = this->src_mesh->axis0->size(), n1 = this->src_mesh->axis1->size();
    Vec<2> p = this->flags.wrap(this->dst_mesh->at(index));

    int i0 = this->src_mesh->axis0->findIndex(p.c0),
        i1 = this->src_mesh->axis1->findIndex(p.c1);

    if (i0 == 0) { ++i0; p.c0 = this->src_mesh->axis0->at(0); }
    else if (i0 == n0) { --i0; p.c0 = this->src_mesh->axis0->at(i0); }
    if (i1 == 0) { ++i1; p.c1 = this->src_mesh->axis1->at(0); }
    else if (i1 == n1) { --i1; p.c1 = this->src_mesh->axis1->at(i1); }

    if (n0 > 1 && n1 > 1) {
        double d0 = this->src_mesh->axis0->at(i0) - this->src_mesh->axis0->at(i0-1),
               d1 = this->src_mesh->axis1->at(i1) - this->src_mesh->axis1->at(i1-1);
        double x0 = (p.c0 - this->src_mesh->axis0->at(i0-1)) / d0,
               x1 = (p.c1 - this->src_mesh->axis1->at(i1-1)) / d1;
        // Hermite 3rd order spline polynomials (in Horner form)
        double hl = ( 2.*x0 - 3.) * x0*x0 + 1.,
               hr = (-2.*x0 + 3.) * x0*x0,
               gl = ((x0 - 2.) * x0 + 1.) * x0 * d0,
               gr = (x0 - 1.) * x0 * x0 * d0,
               hb = ( 2.*x1 - 3.) * x1*x1 + 1.,
               ht = (-2.*x1 + 3.) * x1*x1,
               gb = ((x1 - 2.) * x1 + 1.) * x1 * d1,
               gt = (x1 - 1.) * x1 * x1 * d1;
        int ilb = this->src_mesh->index(i0-1, i1-1),
            ilt = this->src_mesh->index(i0-1, i1),
            irb = this->src_mesh->index(i0, i1-1),
            irt = this->src_mesh->index(i0, i1);
        return this->flags.postprocess(this->dst_mesh->at(index),
            hl * (hb * this->src_vec[ilb] + ht * this->src_vec[ilt]) + hr * (hb * this->src_vec[irb] + ht * this->src_vec[irt]) +
            hb * (gl * diff0[ilb] + gr * diff0[irb]) + ht * (gl * diff0[ilt] + gr * diff0[irt]) +
            hl * (gb * diff1[ilb] + gt * diff1[ilt]) + hr * (gb * diff1[irb] + gt * diff1[irt]));
    } else if (n0 > 1) {
        double d = this->src_mesh->axis0->at(i0) - this->src_mesh->axis0->at(i0-1);
        double x = (p.c0 - this->src_mesh->axis0->at(i0-1)) / d;
        // Hermite 3rd order spline polynomials (in Horner form)
        double ha = ( 2.*x - 3.) * x*x + 1.,
               hb = (-2.*x + 3.) * x*x,
               ga = ((x - 2.) * x + 1.) * x * d,
               gb = (x - 1.) * x * x * d;
        return this->flags.postprocess(this->dst_mesh->at(index), ha*this->src_vec[i0-1] + hb*this->src_vec[i0] + ga*diff0[i0-1] + gb*diff0[i0]);
    } else if (n1 > 1) {
        double d = this->src_mesh->axis1->at(i1) - this->src_mesh->axis1->at(i1-1);
        double x = (p.c1 - this->src_mesh->axis1->at(i1-1)) / d;
        // Hermite 3rd order spline polynomials (in Horner form)
        double ha = ( 2.*x - 3.) * x*x + 1.,
               hb = (-2.*x + 3.) * x*x,
               ga = ((x - 2.) * x + 1.) * x * d,
               gb = (x - 1.) * x * x * d;
        return this->flags.postprocess(this->dst_mesh->at(index), ha*this->src_vec[i1-1] + hb*this->src_vec[i1] + ga*diff1[i1-1] + gb*diff1[i1]);
    }
    return this->flags.postprocess(this->dst_mesh->at(index), this->src_vec[0]);
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
                detail::computeDiffs(diff0.data()+offset, src_mesh->axis0, src_vec.data()+offset, stride0);
            }
        }
    }
    if (n1 > 1) {
        size_t stride1 = src_mesh->index(0, 1, 0);
        for (size_t i2 = 0; i2 < src_mesh->axis2->size(); ++i2) {
            for (size_t i0 = 0; i0 < src_mesh->axis0->size(); ++i0) {
                size_t offset = src_mesh->index(i0, 0, i2);
                detail::computeDiffs(diff1.data()+offset, src_mesh->axis1, src_vec.data()+offset, stride1);
            }
        }
    }
    if (n2 > 1) {
        size_t stride2 = src_mesh->index(0, 0, 1);
        for (size_t i1 = 0; i1 < src_mesh->axis1->size(); ++i1) {
            for (size_t i0 = 0; i0 < src_mesh->axis0->size(); ++i0) {
                size_t offset = src_mesh->index(i0, i1, 0);
                detail::computeDiffs(diff2.data()+offset, src_mesh->axis2, src_vec.data()+offset, stride2);
            }
        }
    }
}

template <typename DstT, typename SrcT>
DstT SplineRect3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    const int n0 = this->src_mesh->axis0->size(), n1 = this->src_mesh->axis1->size(), n2 = this->src_mesh->axis2->size();
    Vec<3> p = this->flags.wrap(this->dst_mesh->at(index));

    int i0 = this->src_mesh->axis0->findIndex(p.c0),
        i1 = this->src_mesh->axis1->findIndex(p.c1),
        i2 = this->src_mesh->axis2->findIndex(p.c2);

    if (i0 == 0) { ++i0; p.c0 = this->src_mesh->axis0->at(0); }
    if (i0 == n0) { --i0; p.c0 = this->src_mesh->axis0->at(i0); }
    if (i1 == 0) { ++i1; p.c1 = this->src_mesh->axis1->at(0); }
    if (i1 == n1) { --i1; p.c1 = this->src_mesh->axis1->at(i1); }
    if (i2 == 0) { ++i2; p.c2 = this->src_mesh->axis2->at(0); }
    if (i2 == n2) { --i2; p.c2 = this->src_mesh->axis2->at(i2); }

    double d0 = this->src_mesh->axis0->at(i0) - this->src_mesh->axis0->at(i0? i0-1 : 0),
           d1 = this->src_mesh->axis1->at(i1) - this->src_mesh->axis1->at(i1? i1-1 : 0),
           d2 = this->src_mesh->axis2->at(i2) - this->src_mesh->axis2->at(i2? i2-1 : 0);
    double x0 = i0? (p.c0 - this->src_mesh->axis0->at(i0-1)) / d0 : 0.,
           x1 = i1? (p.c1 - this->src_mesh->axis1->at(i1-1)) / d1 : 0.,
           x2 = i2? (p.c2 - this->src_mesh->axis2->at(i2-1)) / d2 : 0.;
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
    int illl = this->src_mesh->index(i0?i0-1:0, i1?i1-1:0, i2?i2-1:0),
        illh = this->src_mesh->index(i0?i0-1:0, i1?i1-1:0, i2),
        ilhl = this->src_mesh->index(i0?i0-1:0, i1, i2?i2-1:0),
        ilhh = this->src_mesh->index(i0?i0-1:0, i1, i2),
        ihll = this->src_mesh->index(i0, i1?i1-1:0, i2?i2-1:0),
        ihlh = this->src_mesh->index(i0, i1?i1-1:0, i2),
        ihhl = this->src_mesh->index(i0, i1, i2?i2-1:0),
        ihhh = this->src_mesh->index(i0, i1, i2);
    return this->flags.postprocess(this->dst_mesh->at(index),
        h0l * h1l * h2l * this->src_vec[illl] +
        h0l * h1l * h2h * this->src_vec[illh] +
        h0l * h1h * h2l * this->src_vec[ilhl] +
        h0l * h1h * h2h * this->src_vec[ilhh] +
        h0h * h1l * h2l * this->src_vec[ihll] +
        h0h * h1l * h2h * this->src_vec[ihlh] +
        h0h * h1h * h2l * this->src_vec[ihhl] +
        h0h * h1h * h2h * this->src_vec[ihhh] +
        (g0l * diff0[illl]) * h1l * h2l + h0l * (g1l * diff1[illl]) * h2l + h0l * h1l * (g2l * diff2[illl]) +
        (g0l * diff0[illh]) * h1l * h2h + h0l * (g1l * diff1[illh]) * h2h + h0l * h1l * (g2h * diff2[illh]) +
        (g0l * diff0[ilhl]) * h1h * h2l + h0l * (g1h * diff1[ilhl]) * h2l + h0l * h1h * (g2l * diff2[ilhl]) +
        (g0l * diff0[ilhh]) * h1h * h2h + h0l * (g1h * diff1[ilhh]) * h2h + h0l * h1h * (g2h * diff2[ilhh]) +
        (g0h * diff0[ihll]) * h1l * h2l + h0h * (g1l * diff1[ihll]) * h2l + h0h * h1l * (g2l * diff2[ihll]) +
        (g0h * diff0[ihlh]) * h1l * h2h + h0h * (g1l * diff1[ihlh]) * h2h + h0h * h1l * (g2h * diff2[ihlh]) +
        (g0h * diff0[ihhl]) * h1h * h2l + h0h * (g1h * diff1[ihhl]) * h2l + h0h * h1h * (g2l * diff2[ihhl]) +
        (g0h * diff0[ihhh]) * h1h * h2h + h0h * (g1h * diff1[ihhh]) * h2h + h0h * h1h * (g2h * diff2[ihhh])
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
