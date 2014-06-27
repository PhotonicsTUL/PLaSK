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
                                                               const shared_ptr<const MeshD<2>>& dst_mesh):
    InterpolatedLazyDataImpl<DstT, RectangularMesh<2>, const SrcT>(src_mesh, src_vec, dst_mesh),
    diff0(src_mesh->size()), diff1(src_mesh->size())
{
    const int n0 = src_mesh->axis0->size(), n1 = src_mesh->axis1->size();

    if (n0 == 0 || n1 == 0) throw BadMesh("interpolate", "Source mesh empty");

    size_t stride0 = src_mesh->index(1, 0),
           stride1 = src_mesh->index(0, 1);

    if (n0 > 1)
        for (size_t i1 = 0, i = 0; i1 < src_mesh->axis1->size(); ++i1, i += stride1) {
            detail::computeDiffs(diff0.data()+i, src_mesh->axis0, src_vec.data()+i, stride0);
        }
    else
        std::fill(diff0.begin(), diff0.end(), 0.*SrcT());
    if (n1 > 1)
        for (size_t i0 = 0, i = 0; i0 < src_mesh->axis0->size(); ++i0, i += stride0) {
            detail::computeDiffs(diff1.data()+i, src_mesh->axis1, src_vec.data()+i, stride1);
        }
    else
        std::fill(diff1.begin(), diff1.end(), 0.*SrcT());
}

template <typename DstT, typename SrcT>
DstT SplineRect2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    const int n0 = this->src_mesh->axis0->size(), n1 = this->src_mesh->axis1->size();
    Vec<2> p = this->dst_mesh->at(index);

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
        return hl * (hb * this->src_vec[ilb] + ht * this->src_vec[ilt]) + hr * (hb * this->src_vec[irb] + ht * this->src_vec[irt]) +
            hb * (gl * diff0[ilb] + gr * diff0[irb]) + ht * (gl * diff0[ilt] + gr * diff0[irt]) +
            hl * (gb * diff1[ilb] + gt * diff1[ilt]) + hr * (gb * diff1[irb] + gt * diff1[irt]);
    } else if (n0 > 1) {
        double d = this->src_mesh->axis0->at(i0) - this->src_mesh->axis0->at(i0-1);
        double x = (p.c0 - this->src_mesh->axis0->at(i0-1)) / d;
        // Hermite 3rd order spline polynomials (in Horner form)
        double ha = ( 2.*x - 3.) * x*x + 1.,
               hb = (-2.*x + 3.) * x*x,
               ga = ((x - 2.) * x + 1.) * x * d,
               gb = (x - 1.) * x * x * d;
        return ha*this->src_vec[i0-1] + hb*this->src_vec[i0] + ga*diff0[i0-1] + gb*diff0[i0];
    } else if (n1 > 1) {
        double d = this->src_mesh->axis1->at(i1) - this->src_mesh->axis1->at(i1-1);
        double x = (p.c1 - this->src_mesh->axis1->at(i1-1)) / d;
        // Hermite 3rd order spline polynomials (in Horner form)
        double ha = ( 2.*x - 3.) * x*x + 1.,
               hb = (-2.*x + 3.) * x*x,
               ga = ((x - 2.) * x + 1.) * x * d,
               gb = (x - 1.) * x * x * d;
        return ha*this->src_vec[i1-1] + hb*this->src_vec[i1] + ga*diff1[i1-1] + gb*diff1[i1];
    }
    return this->src_vec[0];
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

} // namespace plask
