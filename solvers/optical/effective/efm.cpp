#include "efm.h"
#include "bessel/bessel.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

EffectiveFrequency2DSolver::EffectiveFrequency2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCylindrical, RectilinearMesh2D>(name),
    log_stripe(dataLog<dcomplex, dcomplex>(getId(), "veff", "det")),
    log_value(dataLog<dcomplex, dcomplex>(getId(), "wavelength", "det")),
    have_veffs(false),
    have_fields(false),
    l(0),
    outer_distance(0.1),
    outIntensity(this, &EffectiveFrequency2DSolver::getLightIntenisty) {
    inTemperature = 300.;
    inGain = NAN;
    root.tolx = 1.0e-7;
    root.tolf_min = 1.0e-8;
    root.tolf_max = 1.0e-6;
    root.maxstep = 0.1;
    root.maxiterations = 500;
    striperoot.tolx = 1.0e-5;
    striperoot.tolf_min = 1.0e-7;
    striperoot.tolf_max = 1.0e-6;
    striperoot.maxstep = 0.5;
    striperoot.maxiterations = 500;
}


void EffectiveFrequency2DSolver::loadParam(const std::string& param, XMLReader& reader, Manager&) {
    if (param == "mode") {
        l = reader.getAttribute<unsigned short>("l", l);
    } else if (param == "root") {
            root.tolx = reader.getAttribute<double>("tolx", root.tolx);
            root.tolf_min = reader.getAttribute<double>("tolf_min", root.tolf_min);
            root.tolf_max = reader.getAttribute<double>("tolf_max", root.tolf_max);
            root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
            root.maxiterations = reader.getAttribute<int>("maxiterations", root.maxstep);
    } else if (param == "striperoot") {
            striperoot.tolx = reader.getAttribute<double>("tolx", striperoot.tolx);
            striperoot.tolf_min = reader.getAttribute<double>("tolf_min", striperoot.tolf_min);
            striperoot.tolf_max = reader.getAttribute<double>("tolf_max", striperoot.tolf_max);
            striperoot.maxstep = reader.getAttribute<double>("maxstep", striperoot.maxstep);
            striperoot.maxiterations = reader.getAttribute<int>("maxiterations", striperoot.maxiterations);
    } else if (param == "outer") {
            outer_distance = reader.requireAttribute<double>("distance");
    } else
        throw XMLUnexpectedElementException(reader, "<geometry>, <mesh>, <mode>, <striperoot>, <root>, or <outer>", param);
    reader.requireTagEnd();
}

dcomplex EffectiveFrequency2DSolver::computeMode(dcomplex lambda)
{
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = %1%", str(lambda));
    k0 = 2e3*M_PI / lambda;
    stageOne();
    dcomplex result = RootDigger(*this, [this](const dcomplex& v){return this->detS(v);}, log_value, root).getSolution(0.);
    outWavelength = result;
    outWavelength.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return result;
}



std::vector<dcomplex> EffectiveFrequency2DSolver::findModes(dcomplex lambda1, dcomplex lambda2, unsigned steps, unsigned nummodes)
{
    writelog(LOG_INFO, "Searching for the modes for wavelength between %1% and %2%", str(lambda1), str(lambda2));
    k0 = 4e3*M_PI / (lambda1 + lambda2);
    stageOne();
    dcomplex v1 = (k0 - (2e3*M_PI/lambda1)) / k0;
    dcomplex v2 = (k0 - (2e3*M_PI/lambda2)) / k0;
    return RootDigger(*this, [this](const dcomplex& v){return this->detS(v);}, log_value, root)
            .searchSolutions(v1, v2, steps, 0, nummodes);
}

std::vector<dcomplex> EffectiveFrequency2DSolver::findModesMap(dcomplex lambda1, dcomplex lambda2, unsigned steps)
{
    writelog(LOG_INFO, "Searching for the approximate modes for wavelength between %1% and %2%", str(lambda1), str(lambda2));
    k0 = 4e3*M_PI / (lambda1 + lambda2);
    stageOne();
    dcomplex v1 = (k0 - (2e3*M_PI/lambda1)) / k0;
    dcomplex v2 = (k0 - (2e3*M_PI/lambda2)) / k0;
    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, root)
            .findMap(v1, v2, steps, 0);
}


void EffectiveFrequency2DSolver::onInitialize()
{
    // Set default mesh
    if (!mesh) setSimpleMesh();

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(mesh->tran().size(), std::vector<dcomplex>(mesh->up().size()+1));
    ngCache.assign(mesh->tran().size(), std::vector<dcomplex>(mesh->up().size()+1));
    veffs.resize(mesh->tran().size());
    nng.resize(mesh->tran().size());
}


void EffectiveFrequency2DSolver::onInvalidate()
{
    outWavelength.invalidate();
    have_fields = false;
    outWavelength.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/

/* It would probably be better to use S-matrix method, but for simplicity we use T-matrix */

using namespace Eigen;

void EffectiveFrequency2DSolver::onBeginCalculation(bool fresh)
{
    // Some additional checks
    for (auto x: mesh->axis0) {
        if (x < 0.) throw BadMesh(getId(), "for cylindrical geometry no radial points can be negative");
    }
    if (abs(mesh->axis0[0]) > SMALL) throw BadMesh(getId(), "radial mesh must start from zero");

    size_t rsize = mesh->tran().size();
    size_t zsize = mesh->up().size() + 1;

    if (fresh || inTemperature.changed || inGain.changed || inGainSlope.changed || l != old_l) { // We need to update something

        old_l = l;

        double lam = real(2e3*M_PI / k0);

        writelog(LOG_DEBUG, "Updating refractive indices cache");
        auto temp = inTemperature(*mesh);

        auto midmesh = mesh->getMidpointsMesh();
        auto gain = inGain(midmesh);
        auto gain_slope = inGainSlope.optional(midmesh);

        double h = lam * 1e6*SMALL;
        double lam1 = lam - h, lam2 = lam + h;

        for (size_t ix = 0; ix != rsize; ++ix) {
            size_t tx1;
            double x0, x1;
            x0 = mesh->axis0[ix];
            if (ix < rsize-1) { tx1 = ix+1; x1 = mesh->axis0[tx1]; } else { tx1 = rsize-1; x1 = mesh->axis0[tx1] + 2.*outer_distance; }
            for (size_t iy = 0; iy != zsize; ++iy) {
                size_t ty0, ty1;
                double y0, y1;
                if (iy > 0) { ty0 = iy-1; y0 = mesh->axis1[ty0]; } else { ty0 = 0; y0 = mesh->axis1[ty0] - 2.*outer_distance; }
                if (iy < zsize-1) { ty1 = iy; y1 = mesh->axis1[ty1]; } else { ty1 = zsize-2; y1 = mesh->axis1[ty1] + 2.*outer_distance; }

                double T = 0.25 * ( temp[mesh->index(ix,ty0)] + temp[mesh->index(ix,ty1)] +
                                    temp[mesh->index(tx1,ty0)] + temp[mesh->index(tx1,ty1)] );

                auto material = geometry->getMaterial(0.25 * (vec(x0,y0) + vec(x0,y1) + vec(x1,y0) + vec(x1,y1)));

                // Nr = nr + 1/(4π) λ g
                // Ng = Nr - λ dN/dλ = Nr - λ dn/dλ - 1/(4π) λ^2 dg/dλ
                nrCache[ix][iy] = material->Nr(lam, T);
                ngCache[ix][iy] = nrCache[ix][iy] - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) / (2*h);
                double g = (ix == rsize-1 || iy == 0 || iy == zsize-1)? NAN : gain[midmesh.index(ix, iy-1)];
                if (gain_slope) {
                    double gs = (ix == rsize-1 || iy == 0 || iy == zsize-1)? NAN : (*gain_slope)[midmesh.index(ix, iy-1)];
                    ngCache[ix][iy] -= dcomplex(0., (std::isnan(gs))? 0. : lam*lam * gs * 7.95774715459e-09);
                }
                nrCache[ix][iy] += dcomplex(0., std::isnan(g)? 0. : lam * g * 7.95774715459e-09);

            }
        }

        have_veffs = false;
    }
}

void EffectiveFrequency2DSolver::stageOne()
{
    initCalculation();

    if (!have_veffs) {

        // Compute effective indices for all stripes
        // TODO: start form the stripe with highest refractive index and use effective index of adjacent stripe to find the new one
        for (size_t i = 0; i != nrCache.size(); ++i) {

            writelog(LOG_DETAIL, "Computing effective frequency for vertical stripe %1%", i);
            std::stringstream nrgs; for (auto nr = nrCache[i].begin(), ng = ngCache[i].begin(); nr != nrCache[i].end(); ++nr, ++ng)
                nrgs << ", " << str(*nr) << "/" << str(*ng);
            writelog(LOG_DEBUG, "nR/nG[%1%] = [%2% ]", i, nrgs.str().substr(1));

            dcomplex same_nr = nrCache[i].front();
            dcomplex same_ng = ngCache[i].front();
            bool all_the_same = true;
            for (auto nr = nrCache[i].begin(), ng = ngCache[i].begin(); nr != nrCache[i].end(); ++nr, ++ng)
                if (*nr != same_nr || *ng != same_ng) { all_the_same = false; break; }
            if (all_the_same) {
                veffs[i] = 1.; // TODO make sure this is so!
                nng[i] = same_nr * same_ng;
            } else {
                RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i],ngCache[i]);}, log_stripe, striperoot);
                veffs[i] = rootdigger.getSolution(0.);
                computeStripeNNg(i);
            }
        }

        std::stringstream nrs; for (size_t i = 0; i < veffs.size(); ++i) nrs << ", " << str(veffs[i]);
        writelog(LOG_DEBUG, "stripes veff = [%1% ]", nrs.str().substr(1));

        have_veffs = true;
    }
}


dcomplex EffectiveFrequency2DSolver::detS1(const dcomplex& x, const std::vector<dcomplex>& NR, const std::vector<dcomplex>& NG)
{
    size_t N = NR.size();

    std::vector<dcomplex> beta(N);
    for (size_t i = 0; i < N; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - x * NR[i]*NG[i]);
        if (real(beta[i]) < 0.) beta[i] = -beta[i];  // TODO verify this condition; in general it should consider really outgoing waves
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex n = 0.5 * beta[i]/beta[i+1];
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };

    Vector2cd E; E << 0., 1.;
    E = fresnel(0) * E;

    for (size_t i = 1; i < N-1; ++i) {
        double d = mesh->axis1[i] - mesh->axis1[i-1];
        dcomplex phas = exp(-I * beta[i] * d);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        E = P * E;
        E = fresnel(i) * E;
    }

    return E[1];
}

void EffectiveFrequency2DSolver::computeStripeNNg(size_t stripe)
{
    size_t N = nrCache[stripe].size();
    dcomplex veff = veffs[stripe];

    nng[stripe] = 0.;
    double sum = 0.;

    std::vector<dcomplex> beta(N);
    for (size_t i = 0; i < N; ++i) {
        beta[i] = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - veff * nrCache[stripe][i]*ngCache[stripe][i]);
        if (real(beta[i]) < 0.) beta[i] = -beta[i];  // TODO verify this condition; in general it should consider really outgoing waves
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex n = 0.5 * beta[i]/beta[i+1];
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };

    Vector2cd E; E << 0., 1.;
    E = fresnel(0) * E;

    for (size_t i = 1; i < N-1; ++i) {
        double d = mesh->axis1[i] - mesh->axis1[i-1];
        dcomplex phas = exp(-I * beta[i] * d);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        E = P * E;
        double f = real(E[0] * conj(E[0])) + real(E[1] * conj(E[1]));
        nng[stripe] = f * nrCache[stripe][i] * ngCache[stripe][i];
        sum += f;
        E = fresnel(i) * E;
    }

    nng[stripe] /= sum;
}


Matrix2cd EffectiveFrequency2DSolver::getMatrix(dcomplex v, size_t i)
{
    double r = mesh->axis0[i];

    dcomplex x1 = r * k0 * sqrt(nng[i-1] * (veffs[i-1]-v));
    if (real(x1) < 0.) x1 = -x1;

    dcomplex x2 = r * k0 * sqrt(nng[i] * (veffs[i]-v));
    if (real(x2) < 0.) x2 = -x2;

    Matrix2cd A, B;

//     dcomplex J1 = BesselJ(l, x1), H1 = BesselH(l, x1), J1p = BesselJ(l+1, x1), H1p = BesselH(l+1, x1);
//     dcomplex J2 = BesselJ(l, x2), H2 = BesselH(l, x2), J2p = BesselJ(l+1, x2), H2p = BesselH(l+1, x2);
//
//     A << J1,                   H1,
//          l * J1 - x1 * J1p,    l * J1 - x1 * J1p;
//
//     B << J2,                   H2,
//          l * J2 - x2 * J2p,    l * J2 - x2 * J2p;

    return B.inverse() * A;
}


dcomplex EffectiveFrequency2DSolver::detS(const dcomplex& v)
{
//     Matrix2cd T = getMatrix(x);
//     // Rn = | T00 T01 | R0
//     // Ln = | T10 T11 | L0
//     if (symmetry == SYMMETRY_POSITIVE) return T(1,0) + T(1,1);      // R0 = L0   Ln = 0
//     else if (symmetry == SYMMETRY_NEGATIVE) return T(1,0) - T(1,1); // R0 = -L0  Ln = 0
//     else return T(1,1);                                             // R0 = 0    Ln = 0
}



const DataVector<double> EffectiveFrequency2DSolver::getLightIntenisty(const MeshD<2>& dst_mesh, InterpolationMethod)
{
//     if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);
//
//     dcomplex lambda = outNeff();
//
//     writelog(LOG_INFO, "Computing field distribution for Neff = %1%", str(lambda));
//
//     size_t Nx = mesh->tran().size()+1;
//     std::vector<dcomplex> betax(Nx);
//     for (size_t i = 0; i < Nx; ++i) {
//         betax[i] = k0 * sqrt(veffs[i]*veffs[i] - lambda*lambda);
//         if (imag(betax[i]) > 0.) betax[i] = -betax[i];
//     }
//     if (!have_fields) {
//         auto fresnelX = [&](size_t i) -> Matrix2cd {
//             dcomplex f =  (polarization==TE)? veffs[i]/veffs[i+1] :  1.;
//             dcomplex n = 0.5 * betax[i+1]/betax[i] * f*f;
//             Matrix2cd M; M << (0.5+n), (0.5-n),
//                             (0.5-n), (0.5+n);
//             return M;
//         };
//         fieldR.resize(Nx);
//         fieldR[Nx-1] << 1., 0;
//         fieldWeights.resize(Nx);
//         fieldWeights[Nx-1] = 0.;
//         for (ptrdiff_t i = Nx-2; i >= 0; --i) {
//             fieldR[i].noalias() = fresnelX(i) * fieldR[i+1];
//             double d = (symmetry == NO_SYMMETRY)? mesh->tran()[i] - mesh->tran()[max(int(i)-1, 0)] :
//                        (i == 0)? mesh->tran()[0] : mesh->tran()[i] - mesh->tran()[i-1];
//             dcomplex b = betax[i];
//             dcomplex phas = exp(- I * b * d);
//             DiagonalMatrix<dcomplex, 2> P;
//             P.diagonal() << 1./phas, phas;  // we propagate backward
//             fieldR[i] = P * fieldR[i];
//             // Compute density of the field is stored in the i-th layer
//             dcomplex w_ff, w_bb, w_fb, w_bf;
//             if (d != 0.) {
//                 if (imag(b) != 0) { dcomplex bb = b - conj(b);
//                     w_ff = (exp(-I*d*bb) - 1.) / bb;
//                     w_bb = (exp(+I*d*bb) - 1.) / bb;
//                 } else w_ff = w_bb = dcomplex(0., -d);
//                 if (real(b) != 0) { dcomplex bb = b + conj(b);
//                     w_fb = (exp(-I*d*bb) - 1.) / bb;
//                     w_bf = (exp(+I*d*bb) - 1.) / bb;
//                 } else w_ff = w_bb = dcomplex(0., -d);
//                 fieldWeights[i] = -imag(  fieldR[i][0] * conj(fieldR[i][0]) * w_ff
//                                         - fieldR[i][1] * conj(fieldR[i][1]) * w_bb
//                                         + fieldR[i][0] * conj(fieldR[i][1]) * w_fb
//                                         - fieldR[i][1] * conj(fieldR[i][0]) * w_bb);
//             } else {
//                 fieldWeights[i] = 0.;
//             }
//         }
//         double sumw = 0; for (const double& w: fieldWeights) sumw += w;
//         double factor = 1./sumw; for (double& w: fieldWeights) w *= factor;
//         std::stringstream weightss; for (size_t i = xbegin; i < Nx; ++i) weightss << ", " << str(fieldWeights[i]);
//         writelog(LOG_DEBUG, "field confinement in stripes = [%1% ]", weightss.str().substr(1));
//     }
//
//     size_t Ny = mesh->up().size()+1;
//     size_t mid_x = std::max_element(fieldWeights.begin(), fieldWeights.end()) - fieldWeights.begin();
//     // double max_val = 0.;
//     // for (size_t i = 1; i != Nx; ++i) { // Find stripe with maximum weight that has non-constant refractive indices
//     //     if (fieldWeights[i] > max_val) {
//     //         dcomplex same_val = nrCache[i].front(); bool all_the_same = true;
//     //         for (auto n: nrCache[i]) if (n != same_val) { all_the_same = false; break; }
//     //         if (!all_the_same) {
//     //             max_val = fieldWeights[i];
//     //             mid_x = i;
//     //         }
//     //     }
//     // }
//     writelog(LOG_DETAIL, "Vertical field distribution taken from stripe %1%", mid_x-xbegin);
//     std::vector<dcomplex> betay(Ny);
//     bool all_the_same = true; dcomplex same_n = nrCache[mid_x][0];
//     for (const dcomplex& n: nrCache[mid_x]) if (n != same_n) { all_the_same = false; break; }
//     if (all_the_same) {
//         betay.assign(Ny, 0.);
//     } else {
//         for (size_t i = 0; i < Ny; ++i) {
//             betay[i] = k0 * sqrt(nrCache[mid_x][i]*nrCache[mid_x][i] - veffs[mid_x]*veffs[mid_x]);
//             if (imag(betay[i]) > 0.) betay[i] = -betay[i];
//         }
//     }
//     if (!have_fields) {
//         if (all_the_same) {
//             fieldZ.assign(Ny, 0.5 * Vector2cd::Ones(2));
//         } else {
//             auto fresnelY = [&](size_t i) -> Matrix2cd {
//                 dcomplex f =  (polarization==TM)? nrCache[mid_x][i]/nrCache[mid_x][i+1] :  1.;
//                 dcomplex n = 0.5 * betay[i+1]/betay[i] * f*f;
//                 Matrix2cd M; M << (0.5+n), (0.5-n),
//                                 (0.5-n), (0.5+n);
//                 return M;
//             };
//             fieldZ.resize(Ny);
//             fieldZ[Ny-1] << 1., 0;
//             for (ptrdiff_t i = Ny-2; i >= 0; --i) {
//                 fieldZ[i].noalias() = fresnelY(i) * fieldZ[i+1];
//                 double d = mesh->up()[i] - mesh->up()[max(int(i)-1, 0)];
//                 dcomplex phas = exp(- I * betay[i] * d);
//                 DiagonalMatrix<dcomplex, 2> P;
//                 P.diagonal() << 1./phas, phas;  // we propagate backward
//                 fieldZ[i] = P * fieldZ[i];
//             }
//         }
//     }
//
//     DataVector<double> results(dst_mesh.size());
//     size_t idx = 0;
//
//     if (!getLightIntenisty_Efficient<RectilinearMesh2D>(dst_mesh, results, betax, betay) &&
//         !getLightIntenisty_Efficient<RegularMesh2D>(dst_mesh, results, betax, betay)) {
//
//         for (auto point: dst_mesh) {
//             double x = point.tran;
//             double y = point.up;
//
//             bool negate = false;
//             if (x < 0. && symmetry != NO_SYMMETRY) {
//                 x = -x; if (symmetry == SYMMETRY_NEGATIVE) negate = true;
//             }
//             size_t ix = mesh->tran().findIndex(x);
//             if (ix != 0) x -= mesh->tran()[ix-1];
//             else if (symmetry == NO_SYMMETRY) x -= mesh->tran()[0];
//             dcomplex phasx = exp(- I * betax[ix] * x);
//             dcomplex val = fieldR[ix][0] * phasx + fieldR[ix][1] / phasx;
//             if (negate) val = - val;
//
//             size_t iy = mesh->up().findIndex(y);
//             y -= mesh->up()[max(int(iy)-1, 0)];
//             dcomplex phasy = exp(- I * betay[iy] * y);
//             val *= fieldZ[iy][0] * phasy + fieldZ[iy][1] / phasy;
//
//             results[idx++] = real(abs2(val));
//         }
//
//     }
//
//     // Normalize results to make maximum value equal to one
//     double factor = 1. / *std::max_element(results.begin(), results.end());
//     for (double& val: results) val *= factor;
//
//     return results;
}

template <typename MeshT>
bool EffectiveFrequency2DSolver::getLightIntenisty_Efficient(const plask::MeshD<2>& dst_mesh, DataVector<double>& results,
                                                         const std::vector<dcomplex>& betax, const std::vector<dcomplex>& betay)
{
//     if (dynamic_cast<const MeshT*>(&dst_mesh)) {
//
//         const MeshT& rect_mesh = dynamic_cast<const MeshT&>(dst_mesh);
//
//         std::vector<dcomplex> valx(rect_mesh.tran().size());
//         std::vector<dcomplex> valy(rect_mesh.up().size());
//         size_t idx = 0, idy = 0;
//
//         for (auto x: rect_mesh.tran()) {
//             bool negate = false;
//             if (x < 0. && symmetry != NO_SYMMETRY) {
//                 x = -x; if (symmetry == SYMMETRY_NEGATIVE) negate = true;
//             }
//             size_t ix = mesh->tran().findIndex(x);
//             if (ix != 0) x -= mesh->tran()[ix-1];
//             else if (symmetry == NO_SYMMETRY) x -= mesh->tran()[0];
//             dcomplex phasx = exp(- I * betax[ix] * x);
//             dcomplex val = fieldR[ix][0] * phasx + fieldR[ix][1] / phasx;
//             if (negate) val = - val;
//             valx[idx++] = val;
//         }
//
//         for (auto y: rect_mesh.up()) {
//             size_t iy = mesh->up().findIndex(y);
//             y -= mesh->up()[max(int(iy)-1, 0)];
//             dcomplex phasy = exp(- I * betay[iy] * y);
//             valy[idy++] = fieldZ[iy][0] * phasy + fieldZ[iy][1] / phasy;
//         }
//
//         for (size_t i = 0; i != rect_mesh.size(); ++i) {
//             dcomplex f = valx[rect_mesh.index0(i)] * valy[rect_mesh.index1(i)];
//             results[i] = real(abs2(f));
//         }
//
//         return true;
//     }
//
//     return false;
}


}}} // namespace plask::solvers::effective
