#ifndef PLASK__MESH_RECTANGULARSPLINE_H
#define PLASK__MESH_RECTANGULARSPLINE_H

#include "../math.h"
#include "mesh.h"
#include "interpolation.h"

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
            Hyman<double>::filter(im, real(a), real(b));
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
}

template <typename Mesh1D, typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2,Mesh1D>, SrcT, DstT, INTERPOLATION_SPLINE> {

    static void interpolate(const RectangularMesh<2,Mesh1D>& src_mesh, const DataVector<const SrcT>& src_vec,
                            const plask::MeshD<2>& dst_mesh, DataVector<DstT>& dst_vec) {

        typedef typename std::remove_const<SrcT>::type DataT;

        const int n0 = src_mesh.axis0.size(),
                  n1 = src_mesh.axis1.size();

        if (n0 == 0 || n1 == 0) throw BadMesh("interpolate", "Source mesh empty");

        if (n0 > 1 && n1 > 1) {

            DataVector<DataT> diff0(src_mesh.size()), diff1(src_mesh.size());
            std::vector<bool> have_diff(src_mesh.size(), false);
            for (int di = 0; di < dst_mesh.size(); ++di) {
                Vec<2> p = dst_mesh[di];
                int i0 = src_mesh.axis0.findIndex(p.c0),
                    i1 = src_mesh.axis1.findIndex(p.c1);
                if (i0 == 0) { ++i0; p.c0 = src_mesh.axis0[0]; }
                else if (i0 == n0) { --i0; p.c0 = src_mesh.axis0[i0]; }
                if (i1 == 0) { ++i1; p.c1 = src_mesh.axis1[1]; }
                else if (i1 == n1) { --i1; p.c1 = src_mesh.axis1[i1]; }
                // Compute derivatives if necessary
                for (int j0 = max(i0-1,0); j0 < min(i0+1,n0); ++j0) {
                    for (int j1 = max(i1-1,0); j1 < min(i1+1,n1); ++j1) {
                        const int j = src_mesh.index(j0,j1);
                        if (!have_diff[j]) { // we need to compute derivatives
                            const int j0l = max(j0-1, 0),
                                      j0r = min(j0+1, n0-1),
                                      j1b = max(j1-1, 0),
                                      j1t = min(j1+1, n1-1);
                            const int jl = src_mesh.index(j0l, j1),
                                      jr = src_mesh.index(j0r, j1),
                                      jb = src_mesh.index(j0, j1b),
                                      jt = src_mesh.index(j0, j1t);
                            const double dl = src_mesh.axis0[j0] - src_mesh.axis0[j0l],
                                         dr = src_mesh.axis0[j0r] - src_mesh.axis0[j0],
                                         db = src_mesh.axis1[j1] - src_mesh.axis0[j1b],
                                         dt = src_mesh.axis1[j1t] - src_mesh.axis0[j1];
                            const DataT sl = dl? (src_vec[j] - src_vec[jl]) / dl : 0. * SrcT(),
                                        sr = dr? (src_vec[jr] - src_vec[j]) / dr : 0. * SrcT(),
                                        sb = db? (src_vec[j] - src_vec[jb]) / db : 0. * SrcT(),
                                        st = dt? (src_vec[jt] - src_vec[j]) / dt : 0. * SrcT();
                            // Use parabolic estimation of the derivative
                            diff0[j] = (dl * sr  + dr * sl) / (dl + dr);
                            diff1[j] = (db * st  + dt * sb) / (db + dt);
                            // Hyman filter
                            detail::Hyman<DataT>::filter(diff0[j], sl, sr);
                            detail::Hyman<DataT>::filter(diff1[j], sb, st);
                            // We have computed it
                            have_diff[j] = true;
                        }
                    }
                }

                double d0 = src_mesh.axis0[i0] - src_mesh.axis0[i0-1],
                       d1 = src_mesh.axis1[i1] - src_mesh.axis1[i1-1];
                double x0 = (p.c0 - src_mesh.axis0[i0-1]) / d0,
                       x1 = (p.c1 - src_mesh.axis1[i1-1]) / d1;
                // Hermite 3rd order spline polynomials (in Horner form)
                double hl = ( 2.*x0 - 3.) * x0*x0 + 1.,
                       hr = (-2.*x0 + 3.) * x0*x0,
                       gl = ((x0 - 2.) * x0 + 1.) * x0 * d0,
                       gr = (x0 - 1.) * x0 * x0 * d0,
                       hb = ( 2.*x1 - 3.) * x1*x1 + 1.,
                       ht = (-2.*x1 + 3.) * x1*x1,
                       gb = ((x1 - 2.) * x1 + 1.) * x1 * d1,
                       gt = (x1 - 1.) * x1 * x1 * d1;
                int ilb = src_mesh.index(i0-1, i1-1),
                    ilt = src_mesh.index(i0-1, i1),
                    irb = src_mesh.index(i0, i1-1),
                    irt = src_mesh.index(i0, i1);
                dst_vec[di] = hl * (hb * src_vec[ilb] + ht * src_vec[ilt]) + hr * (hb * src_vec[irb] + ht * src_vec[irt]) +
                              hb * (gl * diff0[ilb] + gr * diff0[irb]) + ht * (gl * diff0[ilt] + gr * diff0[irt]) +
                              hl * (gb * diff1[ilb] + gt * diff1[ilt]) + hr * (gb * diff1[irb] + gt * diff1[irt]);
            }

        } else if (n0 > 1) {

            DataVector<DataT> diff(src_mesh.size());
            std::vector<bool> have_diff(src_mesh.size(), false);
            for (int di = 0; di < dst_mesh.size(); ++di) {
                Vec<2> p = dst_mesh[di];
                int i = src_mesh.axis0.findIndex(p.c0);
                if (i == 0) { ++i; p.c0 = src_mesh.axis0[0]; }
                else if (i == n0) { --i; p.c0 = src_mesh.axis0[i]; }
                // Compute derivatives if necessary
                for (int j = max(i-1,0); j < min(i+1,n0); ++j) {
                    if (!have_diff[j]) { // we need to compute derivatives
                        const int ja = max(j-1, 0),
                                  jb = min(j+1, n0-1);
                        const double da = src_mesh.axis0[j] - src_mesh.axis0[ja],
                                     db = src_mesh.axis0[jb] - src_mesh.axis0[j];
                        const DataT sa = da? (src_vec[j] - src_vec[ja]) / da : 0. * SrcT(),
                                    sb = db? (src_vec[jb] - src_vec[j]) / db : 0. * SrcT();
                        // Use parabolic estimation of the derivative
                        diff[j] = (da * sb  + db * sa) / (da + db);
                        // Hyman filter
                        detail::Hyman<DataT>::filter(diff[j], sa, sb);
                        // We have computed it
                        have_diff[j] = true;
                    }
                }
                double d = src_mesh.axis0[i] - src_mesh.axis0[i-1];
                double x = (p.c0 - src_mesh.axis0[i-1]) / d;
                // Hermite 3rd order spline polynomials (in Horner form)
                double ha = ( 2.*x - 3.) * x*x + 1.,
                       hb = (-2.*x + 3.) * x*x,
                       ga = ((x - 2.) * x + 1.) * x * d,
                       gb = (x - 1.) * x * x * d;
                dst_vec[di] = ha*src_vec[i-1] + hb*src_vec[i] + ga*diff[i-1] + gb*diff[i];
            }

        } else if (n1 > 1) {

            DataVector<DataT> diff(src_mesh.size());
            std::vector<bool> have_diff(src_mesh.size(), false);
            for (int di = 0; di < dst_mesh.size(); ++di) {
                Vec<2> p = dst_mesh[di];
                int i = src_mesh.axis1.findIndex(p.c1);
                if (i == 0) { ++i; p.c1 = src_mesh.axis1[0]; }
                else if (i == n1) { --i; p.c1 = src_mesh.axis1[i]; }
                // Compute derivatives if necessary
                for (int j = max(i-1,0); j < min(i+1,n1); ++j) {
                    if (!have_diff[j]) { // we need to compute derivatives
                        const int ja = max(j-1, 0),
                                  jb = min(j+1, n1-1);
                        const double da = src_mesh.axis1[j] - src_mesh.axis1[ja],
                                     db = src_mesh.axis1[jb] - src_mesh.axis1[j];
                        const DataT sa = da? (src_vec[j] - src_vec[ja]) / da : 0. * SrcT(),
                                    sb = db? (src_vec[jb] - src_vec[j]) / db : 0. * SrcT();
                        // Use parabolic estimation of the derivative
                        diff[j] = (da * sb  + db * sa) / (da + db);
                        // Hyman filter
                        detail::Hyman<DataT>::filter(diff[j], sa, sb);
                        // We have computed it
                        have_diff[j] = true;
                    }
                }
                double d = src_mesh.axis1[i] - src_mesh.axis1[i-1];
                double x = (p.c1 - src_mesh.axis1[i-1]) / d;
                // Hermite 3rd order spline polynomials (in Horner form)
                double ha = ( 2.*x - 3.) * x*x + 1.,
                       hb = (-2.*x + 3.) * x*x,
                       ga = ((x - 2.) * x + 1.) * x * d,
                       gb = (x - 1.) * x * x * d;
                dst_vec[di] = ha*src_vec[i-1] + hb*src_vec[i] + ga*diff[i-1] + gb*diff[i];
            }

        } else { // there is only one point
            std::fill_n(dst_vec.data(), dst_vec.size(), src_vec[0]);
        }
    }

};


} // namespace plask

#endif // PLASK__MESH_RECTANGULARSPLINE_H
