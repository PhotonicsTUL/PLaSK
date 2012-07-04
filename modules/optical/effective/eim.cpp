#include "eim.h"

#include <plask/mesh/regular.h>

using plask::dcomplex;

namespace plask { namespace modules { namespace effective {

EffectiveIndex2DModule::EffectiveIndex2DModule() :
    log_stripe(dataLog<dcomplex, dcomplex>(getId(), "neff", "det")),
    log_value(dataLog<dcomplex, dcomplex>(getId(), "Neff", "det")),
    have_stripeNeffs(false),
    have_fields(false),
    old_polarization(TE),
    polarization(TE),
    symmetry(NO_SYMMETRY),
    outer_distance(0.1),
    tolx(1.0e-07),
    tolf_min(1.0e-8),
    tolf_max(1.0e-6),
    maxstep(0.1),
    maxiterations(500),
    outIntensity(this, &EffectiveIndex2DModule::getLightIntenisty) {
    inTemperature = 300.;
}


dcomplex EffectiveIndex2DModule::computeMode(dcomplex neff)
{
    writelog(LOG_INFO, "Searching for the mode starting from Neff = %1%", str(neff));
    stageOne();
    dcomplex result = RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, tolx, tolf_min, tolf_max, maxstep, maxiterations).getSolution(neff);
    outNeff = result;
    outNeff.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return result;
}



std::vector<dcomplex> EffectiveIndex2DModule::findModes(dcomplex neff1, dcomplex neff2, unsigned steps, unsigned nummodes)
{
    writelog(LOG_INFO, "Searching for the modes for Neff between %1% and %2%", str(neff1), str(neff2));
    stageOne();
    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, tolx, tolf_min, tolf_max, maxstep, maxiterations)
            .searchSolutions(neff1, neff2, steps, 0, nummodes);
}

std::vector<dcomplex> EffectiveIndex2DModule::findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps)
{
    writelog(LOG_INFO, "Searching for the approximate modes for Neff between %1% and %2%", str(neff1), str(neff2));
    stageOne();

    double rdneff = real(neff2 - neff1);
    double rneff1 = real(neff1);
    double steps1 = steps + 1;
    std::vector<double> rpoints(steps+1);
    for (unsigned i = 0; i <= steps; ++i) {
        rpoints[i] = rneff1 + rdneff * i / steps1;
    }

    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, tolf_min, tolf_max, maxstep, maxiterations)
            .findMap(neff1, neff2, steps, 0);
}


void EffectiveIndex2DModule::setMode(dcomplex neff)
{
    if (!initialized) {
        writelog(LOG_WARNING, "Module invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    double det = abs(detS(neff));
    if (det > tolf_max) throw BadInput(getId(), "Provided effective index does not correspond to any mode (det = %1%)", det);
    writelog(LOG_INFO, "Setting current mode to %1%", str(neff));
    outNeff = neff;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}


void EffectiveIndex2DModule::onInitialize()
{
    // Set default mesh
    if (!mesh) setSimpleMesh();

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(mesh->tran().size()+1, std::vector<dcomplex>(mesh->up().size()+1));
    stripeNeffs.resize(mesh->tran().size()+1);
}


void EffectiveIndex2DModule::onInvalidate()
{
    outNeff.invalidate();
    have_fields = false;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/

/* It would probably be better to use S-matrix method, but for simplicity we use T-matrix */

using namespace Eigen;



void EffectiveIndex2DModule::updateCache()
{
    bool updated = initCalculation();

    xbegin = 0;

    // Some additional checks
    if (symmetry == SYMMETRY_POSITIVE || symmetry == SYMMETRY_NEGATIVE) {
        if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
            if (updated) // Make sure we have only positive points
                for (auto x: mesh->c0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
            if (mesh->c0[0] == 0.) xbegin = 1;
        } else {
            writelog(LOG_WARNING, "Symmetry reset to NO_SYMMETRY for non-symmetric geometry.");
            symmetry = NO_SYMMETRY;
        }
    }

    size_t xsize = mesh->tran().size() + 1;
    size_t ysize = mesh->up().size() + 1;

    if (updated || inTemperature.changed || inWavelength.changed || polarization != old_polarization) { // We need to update something

        old_polarization = polarization;

        k0 = 2e3*M_PI / inWavelength();
        double w = real(inWavelength());

        writelog(LOG_DEBUG, "Updating refractive indices cache");
        auto temp = inTemperature(*mesh);
        for (size_t i = xbegin; i != xsize; ++i) {
            size_t tx0, tx1;
            double x0, x1;
            if (i > 0) { tx0 = i-1; x0 = mesh->c0[tx0]; } else { tx0 = 0; x0 = mesh->c0[tx0] - outer_distance; }
            if (i < xsize-1) { tx1 = i; x1 = mesh->c0[tx1]; } else { tx1 = xsize-2; x1 = mesh->c0[tx1] + outer_distance; }
            for (size_t j = 0; j != ysize; ++j) {
                size_t ty0, ty1;
                double y0, y1;
                if (j > 0) { ty0 = j-1; y0 = mesh->c1[ty0]; } else { ty0 = 0; y0 = mesh->c1[ty0] - outer_distance; }
                if (j < ysize-1) { ty1 = j; y1 = mesh->c1[ty1]; } else { ty1 = ysize-2; y1 = mesh->c1[ty1] + outer_distance; }
                double T = 0.25 * ( temp[mesh->index(tx0,ty0)] + temp[mesh->index(tx0,ty1)] +
                                    temp[mesh->index(tx1,ty0)] + temp[mesh->index(tx1,ty1)] );
                nrCache[i][j] = geometry->getMaterial(0.25 * (vec(x0,y0) + vec(x0,y1) + vec(x1,y0) + vec(x1,y1)))->Nr(w, T);
            }
        }
        if (xbegin == 1) nrCache[0] = nrCache[1];

        have_stripeNeffs = false;
    }
}

void EffectiveIndex2DModule::stageOne()
{
    updateCache();

    if (!have_stripeNeffs) {

        // Compute effective indices for all stripes
        // TODO: start form the stripe with highest refractive index and use effective index of adjacent stripe to find the new one
        for (size_t i = xbegin; i != nrCache.size(); ++i) {

            writelog(LOG_DETAIL, "Computing effective index for vertical stripe %1% (polarization %2%)", i-xbegin, (polarization==TE)?"TE":"TM");
            std::stringstream nrs; for (auto nr: nrCache[i]) nrs << ", " << str(nr);
            writelog(LOG_DEBUG, "nR[%1%] = [%2% ]", i-xbegin, nrs.str().substr(1));

            dcomplex same_val = nrCache[i].front();
            bool all_the_same = true;
            for (auto n: nrCache[i]) if (n != same_val) { all_the_same = false; break; }
            if (all_the_same) {
                stripeNeffs[i] = same_val;
            } else {
                RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i]);}, log_stripe, 1e-5, 1e-7, 1e-6, 0.5, maxiterations);
                dcomplex maxn = *std::max_element(nrCache[i].begin(), nrCache[i].end(), [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
                stripeNeffs[i] = rootdigger.getSolution(0.999999*maxn);
                // dcomplex minn = *std::min_element(nrCache[i].begin(), nrCache[i].end(), [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
                // auto map = rootdigger.findMap(0.999999*maxn, 1.000001*minn, initial_stripe_neff_map, 0);
                // stripeNeffs[i] = rootdigger.getSolution(map[0]);
            }
        }
        if (xbegin == 1) stripeNeffs[0] = stripeNeffs[1];

        std::stringstream nrs; for (size_t i = xbegin; i < stripeNeffs.size(); ++i) nrs << ", " << str(stripeNeffs[i]);
        writelog(LOG_DEBUG, "stripes neffs = [%1% ]", nrs.str()/*.substr(1)*/);

        have_stripeNeffs = true;
    }
}


dcomplex EffectiveIndex2DModule::detS1(const plask::dcomplex& x, const std::vector<dcomplex>& NR)
{
    size_t N = NR.size();

    std::vector<dcomplex> beta(N);
    for (size_t i = 0; i < N; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
        if (imag(beta[i]) > 0.) beta[i] = -beta[i];
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TM)? NR[i+1]/NR[i] : 1.;
        dcomplex n = 0.5 * beta[i]/beta[i+1] * f*f;
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };

    Vector2cd E; E << 0., 1.;
    E = fresnel(0) * E;

    for (size_t i = 1; i < N-1; ++i) {
        double d = mesh->c1[i] - mesh->c1[i-1];
        dcomplex phas = exp(-I * beta[i] * d);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        E = P * E;
        E = fresnel(i) * E;
    }

    return E[1];
}


Matrix2cd EffectiveIndex2DModule::getMatrix(dcomplex neff)
{
    // Adjust for mirror losses
    neff = dcomplex(real(neff), imag(neff)-getMirrorLosses(neff));

    size_t N = stripeNeffs.size();

    std::vector<dcomplex> beta(N);
    for (size_t i = xbegin; i < N; ++i) {
        beta[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
        if (imag(beta[i]) > 0.) beta[i] = -beta[i];
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TE)? stripeNeffs[i+1]/stripeNeffs[i] :  1.;
        dcomplex n = 0.5 * beta[i]/beta[i+1] * f*f;
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };


    Matrix2cd T = fresnel(xbegin);

    if (symmetry != NO_SYMMETRY) { // we have symmetry, so begin of the transfer matrix is at the axis
        dcomplex phas = exp(-I * beta[xbegin] * mesh->c0[xbegin]);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        T = T * P;
    }

    for (size_t i = xbegin+1; i < N-1; ++i) {
        double d = mesh->c0[i] - mesh->c0[i-1];
        dcomplex phas = exp(- I * beta[i] * d);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        T = P * T;
        T = fresnel(i) * T;
    }

    return T;
}


dcomplex EffectiveIndex2DModule::detS(const dcomplex& x)
{
    Matrix2cd T = getMatrix(x);
    // Rn = | T00 T01 | R0
    // Ln = | T10 T11 | L0
    if (symmetry == SYMMETRY_POSITIVE) return T(1,0) + T(1,1);      // R0 = L0   Ln = 0
    else if (symmetry == SYMMETRY_NEGATIVE) return T(1,0) - T(1,1); // R0 = -L0  Ln = 0
    else return T(1,1);                                             // R0 = 0    Ln = 0
}



const DataVector<double> EffectiveIndex2DModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod)
{
    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    dcomplex neff = outNeff();

    writelog(LOG_INFO, "Computing field distribution for Neff = %1%", str(neff));

    size_t Nx = mesh->tran().size()+1;
    std::vector<dcomplex> betax(Nx);
    for (size_t i = 0; i < Nx; ++i) {
        betax[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
        if (imag(betax[i]) > 0.) betax[i] = -betax[i];
    }
    if (!have_fields) {
        auto fresnelX = [&](size_t i) -> Matrix2cd {
            dcomplex f =  (polarization==TE)? stripeNeffs[i]/stripeNeffs[i+1] :  1.;
            dcomplex n = 0.5 * betax[i+1]/betax[i] * f*f;
            Matrix2cd M; M << (0.5+n), (0.5-n),
                            (0.5-n), (0.5+n);
            return M;
        };
        fieldX.resize(Nx);
        fieldX[Nx-1] << 1., 0;
        fieldWeights.resize(Nx);
        fieldWeights[Nx-1] = 0.;
        for (ptrdiff_t i = Nx-2; i >= 0; --i) {
            fieldX[i].noalias() = fresnelX(i) * fieldX[i+1];
            double d = (symmetry == NO_SYMMETRY)? mesh->tran()[i] - mesh->tran()[max(int(i)-1, 0)] :
                       (i == 0)? mesh->tran()[0] : mesh->tran()[i] - mesh->tran()[i-1];
            dcomplex b = betax[i];
            dcomplex phas = exp(- I * b * d);
            DiagonalMatrix<dcomplex, 2> P;
            P.diagonal() << 1./phas, phas;  // we propagate backward
            fieldX[i] = P * fieldX[i];
            // Compute density of the field is stored in the i-th layer
            dcomplex w_ff, w_bb, w_fb, w_bf;
            if (d != 0.) {
                if (imag(b) != 0) { dcomplex bb = b - conj(b);
                    w_ff = (exp(-I*d*bb) - 1.) / bb;
                    w_bb = (exp(+I*d*bb) - 1.) / bb;
                } else w_ff = w_bb = dcomplex(0., -d);
                if (real(b) != 0) { dcomplex bb = b + conj(b);
                    w_fb = (exp(-I*d*bb) - 1.) / bb;
                    w_bf = (exp(+I*d*bb) - 1.) / bb;
                } else w_ff = w_bb = dcomplex(0., -d);
                fieldWeights[i] = -imag(  fieldX[i][0] * conj(fieldX[i][0]) * w_ff
                                        - fieldX[i][1] * conj(fieldX[i][1]) * w_bb
                                        + fieldX[i][0] * conj(fieldX[i][1]) * w_fb
                                        - fieldX[i][1] * conj(fieldX[i][0]) * w_bb);
            } else {
                fieldWeights[i] = 0.;
            }
        }
        double sumw = 0; for (const double& w: fieldWeights) sumw += w;
        double factor = 1./sumw; for (double& w: fieldWeights) w *= factor;
        std::stringstream weightss; for (size_t i = xbegin; i < Nx; ++i) weightss << ", " << str(fieldWeights[i]);
        writelog(LOG_DEBUG, "field confinement in stripes = [%1% ]", weightss.str().substr(1));
    }

    size_t Ny = mesh->up().size()+1;
    size_t mid_x = std::max_element(fieldWeights.begin(), fieldWeights.end()) - fieldWeights.begin();
    // double max_val = 0.;
    // for (size_t i = 1; i != Nx; ++i) { // Find stripe with maximum weight that has non-constant refractive indices
    //     if (fieldWeights[i] > max_val) {
    //         dcomplex same_val = nrCache[i].front(); bool all_the_same = true;
    //         for (auto n: nrCache[i]) if (n != same_val) { all_the_same = false; break; }
    //         if (!all_the_same) {
    //             max_val = fieldWeights[i];
    //             mid_x = i;
    //         }
    //     }
    // }
    writelog(LOG_DETAIL, "Vertical field distribution taken from stripe %1%", mid_x-xbegin);
    std::vector<dcomplex> betay(Ny);
    bool all_the_same = true; dcomplex same_n = nrCache[mid_x][0];
    for (const dcomplex& n: nrCache[mid_x]) if (n != same_n) { all_the_same = false; break; }
    if (all_the_same) {
        betay.assign(Ny, 0.);
    } else {
        for (size_t i = 0; i < Ny; ++i) {
            betay[i] = k0 * sqrt(nrCache[mid_x][i]*nrCache[mid_x][i] - stripeNeffs[mid_x]*stripeNeffs[mid_x]);
            if (imag(betay[i]) > 0.) betay[i] = -betay[i];
        }
    }
    if (!have_fields) {
        if (all_the_same) {
            fieldY.assign(Ny, 0.5 * Vector2cd::Ones(2));
        } else {
            auto fresnelY = [&](size_t i) -> Matrix2cd {
                dcomplex f =  (polarization==TM)? nrCache[mid_x][i]/nrCache[mid_x][i+1] :  1.;
                dcomplex n = 0.5 * betay[i+1]/betay[i] * f*f;
                Matrix2cd M; M << (0.5+n), (0.5-n),
                                (0.5-n), (0.5+n);
                return M;
            };
            fieldY.resize(Ny);
            fieldY[Ny-1] << 1., 0;
            for (ptrdiff_t i = Ny-2; i >= 0; --i) {
                fieldY[i].noalias() = fresnelY(i) * fieldY[i+1];
                double d = mesh->up()[i] - mesh->up()[max(int(i)-1, 0)];
                dcomplex phas = exp(- I * betay[i] * d);
                DiagonalMatrix<dcomplex, 2> P;
                P.diagonal() << 1./phas, phas;  // we propagate backward
                fieldY[i] = P * fieldY[i];
            }
        }
    }

    DataVector<double> results(dst_mesh.size());
    size_t idx = 0;

    if (!getLightIntenisty_Efficient<RectilinearMesh2D>(dst_mesh, results, betax, betay) &&
        !getLightIntenisty_Efficient<RegularMesh2D>(dst_mesh, results, betax, betay)) {

        for (auto point: dst_mesh) {
            double x = point.tran;
            double y = point.up;

            bool negate = false;
            if (x < 0. && symmetry != NO_SYMMETRY) {
                x = -x; if (symmetry == SYMMETRY_NEGATIVE) negate = true;
            }
            size_t ix = mesh->tran().findIndex(x);
            if (ix != 0) x -= mesh->tran()[ix-1];
            else if (symmetry == NO_SYMMETRY) x -= mesh->tran()[0];
            dcomplex phasx = exp(- I * betax[ix] * x);
            dcomplex val = fieldX[ix][0] * phasx + fieldX[ix][1] / phasx;
            if (negate) val = - val;

            size_t iy = mesh->up().findIndex(y);
            y -= mesh->up()[max(int(iy)-1, 0)];
            dcomplex phasy = exp(- I * betay[iy] * y);
            val *= fieldY[iy][0] * phasy + fieldY[iy][1] / phasy;

            results[idx++] = real(abs2(val));
        }

    }

    // Normalize results to make maximum value equal to one
    double factor = 1. / *std::max_element(results.begin(), results.end());
    for (double& val: results) val *= factor;

    return results;
}

template <typename MeshT>
bool EffectiveIndex2DModule::getLightIntenisty_Efficient(const plask::Mesh<2>& dst_mesh, DataVector<double>& results,
                                                         const std::vector<dcomplex>& betax, const std::vector<dcomplex>& betay)
{
    if (dynamic_cast<const MeshT*>(&dst_mesh)) {

        const MeshT& rect_mesh = dynamic_cast<const MeshT&>(dst_mesh);

        std::vector<dcomplex> valx(rect_mesh.tran().size());
        std::vector<dcomplex> valy(rect_mesh.up().size());
        size_t idx = 0, idy = 0;

        for (auto x: rect_mesh.tran()) {
            bool negate = false;
            if (x < 0. && symmetry != NO_SYMMETRY) {
                x = -x; if (symmetry == SYMMETRY_NEGATIVE) negate = true;
            }
            size_t ix = mesh->tran().findIndex(x);
            if (ix != 0) x -= mesh->tran()[ix-1];
            else if (symmetry == NO_SYMMETRY) x -= mesh->tran()[0];
            dcomplex phasx = exp(- I * betax[ix] * x);
            dcomplex val = fieldX[ix][0] * phasx + fieldX[ix][1] / phasx;
            if (negate) val = - val;
            valx[idx++] = val;
        }

        for (auto y: rect_mesh.up()) {
            size_t iy = mesh->up().findIndex(y);
            y -= mesh->up()[max(int(iy)-1, 0)];
            dcomplex phasy = exp(- I * betay[iy] * y);
            valy[idy++] = fieldY[iy][0] * phasy + fieldY[iy][1] / phasy;
        }

        for (size_t i = 0; i != rect_mesh.size(); ++i) {
            dcomplex f = valx[rect_mesh.index0(i)] * valy[rect_mesh.index1(i)];
            results[i] = real(abs2(f));
        }

        return true;
    }

    return false;
}


}}} // namespace plask::modules::effective
