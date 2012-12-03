#ifndef PLASK__MESH_RECTANGULARSPLINE_H
#define PLASK__MESH_RECTANGULARSPLINE_H

#include "../math.h"
#include "mesh.h"
#include "interpolation.h"

namespace plask {

namespace detail {
    template <typename T> inline void hyman(T& data, const T& a, const T& b) {
        T lim = 3. * min(abs(a), abs(b));
        if (data > lim) data = lim;
        else if (data < -lim) data = -lim;
    }

    template <> inline void hyman(dcomplex& data, const dcomplex& a, const dcomplex& b) {
        double re = real(data), im = imag(data);
        hyman(re, real(a), real(b));
        hyman(im, real(a), real(b));
        data = dcomplex(re, im);
    }

    template <> inline void hyman(Vec<2,double>& data, const Vec<2,double>& a, const Vec<2,double>& b) {
        hyman(data.c0, a.c0, b.c0);
        hyman(data.c1, a.c1, b.c1);
    }

    template <> inline void hyman(Vec<3,double>& data, const Vec<3,double>& a, const Vec<3,double>& b) {
        hyman(data.c0, a.c0, b.c0);
        hyman(data.c1, a.c1, b.c1);
        hyman(data.c2, a.c2, b.c2);
    }

    template <> inline void hyman(Vec<2,dcomplex>& data, const Vec<2,dcomplex>& a, const Vec<2,dcomplex>& b) {
        hyman(data.c0, a.c0, b.c0);
        hyman(data.c1, a.c1, b.c1);
    }

    template <> inline void hyman(Vec<3,dcomplex>& data, const Vec<3,dcomplex>& a, const Vec<3,dcomplex>& b) {
        hyman(data.c0, a.c0, b.c0);
        hyman(data.c1, a.c1, b.c1);
        hyman(data.c2, a.c2, b.c2);
    }
}

template <typename Mesh1D, typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2,Mesh1D>, SrcT, DstT, INTERPOLATION_HYMAN> {

    static void interpolate(const RectangularMesh<2,Mesh1D>& src_mesh, const DataVector<const SrcT>& src_vec,
                            const plask::MeshD<2>& dst_mesh, DataVector<DstT>& dst_vec) {

        if (src_mesh.axis0.size() > 1 && src_mesh.axis1.size() > 1) {

            DataVector<typename std::remove_const<SrcT>::type> diff0(src_mesh.size(), SrcT(0.)), diff1(src_mesh.size(), SrcT(0.));
            std::vector<bool> have_diff(src_mesh.size());

            for (size_t di = 0; di < dst_mesh.size(); ++di) {
                Vec<2> p = dst_mesh[di];
                size_t i0 = src_mesh.axis0.findIndex(p.c0),
                    i1 = src_mesh.axis1.findIndex(p.c1);

                if (i0 == 0) { ++i0; p.c0 = src_mesh.axis0[0]; }
                else if (i0 == src_mesh.axis0.size()) { --i0; p.c0 = src_mesh.axis0[i0]; }

                if (i1 == 0) { ++i1; p.c1 = src_mesh.axis1[1]; }
                else if (i1 == src_mesh.axis1.size()) { --i1; p.c1 = src_mesh.axis1[i1]; }

                // Compute derivatives if necessary
                for (size_t j0 = max(size_t(1), i0-1); j0 < min(src_mesh.axis0.size()-1, i0+1); ++j0) {
                    for (size_t j1 = max(size_t(1), i1-1); j0 < min(src_mesh.axis1.size()-1, i1+1); ++j1) {
                        size_t j = src_mesh.index(j0,j1);
                        if (!have_diff[j]) { // we need to compute derivatives
                            size_t jl = src_mesh.index(j0-1, j1),
                                jr = src_mesh.index(j0+1, j1),
                                jb = src_mesh.index(j0, j1-1),
                                jt = src_mesh.index(j0, j1+1);
                            double dl = src_mesh.axis0[j0] - src_mesh.axis0[j0-1],
                                dr = src_mesh.axis0[j0+1] - src_mesh.axis0[j0],
                                db = src_mesh.axis1[j1] - src_mesh.axis0[j1-1],
                                dt = src_mesh.axis1[j1+1] - src_mesh.axis0[j1];
                            SrcT sl = (src_vec[j] - src_vec[jl]) / dl,
                                 sr = (src_vec[jr] - src_vec[j]) / dr,
                                 sb = (src_vec[j] - src_vec[jb]) / db,
                                 st = (src_vec[jt] - src_vec[j]) / dt;
                            // Use parabolic estimation of the derivative
                            diff0[j] = (dl * sr  + dr * sl) / (dl + dr);
                            diff1[j] = (db * st  + dt * sb) / (db + dt);
                            // Hyman filter
                            detail::hyman(diff0[j], sl, sr);
                            detail::hyman(diff1[j], sb, st);
                            // We have computed it
                            have_diff[j] = true;
                        }
                    }
                }

                double x0 = (p.c0 - src_mesh.axis0[i0-1]) / src_mesh.axis0[i0] - src_mesh.axis0[i0-1],
                    x1 = (p.c1 - src_mesh.axis1[i1-1]) / src_mesh.axis1[i1] - src_mesh.axis1[i1-1];
                // Hermite 3rd order spline polynomials (in Horner form)
                double hl = ( 2.*x0 - 3.) * x0*x0 + 1.,
                       hr = (-2.*x0 + 3.) * x0*x0,
                       gl = ((x0 - 2.) * x0 + 1.) * x0,
                       gr = (x0 - 1.) * x0 * x0,
                       hb = ( 2.*x1 - 3.) * x1*x1 + 1.,
                       ht = (-2.*x1 + 3.) * x1*x1,
                       gb = ((x1 - 2.) * x1 + 1.) * x1,
                       gt = (x1 - 1.) * x1 * x1;

                size_t ilb = src_mesh.index(i0-1, i1-1),
                    ilt = src_mesh.index(i0-1, i1),
                    irb = src_mesh.index(i0, i1-1),
                    irt = src_mesh.index(i0, i1);

                dst_vec[di] = hl * (hb * src_vec[ilb] + ht * src_vec[ilt]) + hr * (hb * src_vec[irb] + ht * src_vec[irt]) +
                            hb * (gl * diff0[ilb] + gr * diff0[irb]) + ht * (gl * diff0[ilt] + gr * diff0[irt]) +
                            hl * (gb * diff1[ilb] + gt * diff1[ilt]) + hr * (gb * diff1[irb] + gt * diff1[irt]);
            }

        } else {
            throw NotImplemented("Source mesh must have at least two points in each direction");
        }
    }

};


} // namespace plask

#endif // PLASK__MESH_RECTANGULARSPLINE_H