#include "eim.h"

using plask::dcomplex;

namespace plask { namespace modules { namespace eim {

EffectiveIndex2dModule::EffectiveIndex2dModule() :
    log_stripe(dataLog<dcomplex, double>(getId(), "neff", "det")),
    log_value(dataLog<dcomplex, double>(getId(), "Neff", "det")),
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
    outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
    inTemperature = 300.;
}


dcomplex EffectiveIndex2dModule::computeMode(dcomplex neff)
{
    logger(LOG_INFO, "Searching for the mode starting from Neff = %1%", str(neff));
    stageOne();
    dcomplex result = RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, tolx, tolf_min, tolf_max, maxstep, maxiterations).getSolution(neff);
    outNeff = result;
    outNeff.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return result;
}



std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex neff1, dcomplex neff2, unsigned steps, unsigned nummodes)
{
    logger(LOG_INFO, "Searching for the modes for Neff between %1% and %2%", str(neff1), str(neff2));
    stageOne();
    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, tolx, tolf_min, tolf_max, maxstep, maxiterations)
            .searchSolutions(neff1, neff2, steps, 0, nummodes);
}

std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps)
{
    logger(LOG_INFO, "Searching for the approximate modes for Neff between %1% and %2%", str(neff1), str(neff2));
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




void EffectiveIndex2dModule::onInitialize()
{
    // Set default mesh
    if (!mesh) setSimpleMesh();

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(mesh->tran().size()+1, std::vector<dcomplex>(mesh->up().size()+1));
    stripeNeffs.resize(mesh->tran().size()+1);
}


void EffectiveIndex2dModule::onInvalidate()
{
    outNeff.invalidate();
    have_fields = false;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}



void EffectiveIndex2dModule::stageOne()
{
    bool updated = initCalculation();

    xbegin = 0;

    // Some additional checks
    if (symmetry == SYMMETRY_POSITIVE || symmetry == SYMMETRY_NEGATIVE) {
        if (geometry->isSymmetric(CalculationSpace::DIRECTION_TRAN)) {
            if (updated) // Make sure we have only positive points
                for (auto x: mesh->c0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
            if (mesh->c0[0] == 0.) xbegin = 1;
        } else {
            logger(LOG_WARNING, "Symmetry reset to NO_SYMMETRY for non-symmetric geometry.");
            symmetry = NO_SYMMETRY;
        }
    }

    size_t xsize = mesh->tran().size() + 1;
    size_t ysize = mesh->up().size() + 1;

    size_t N = stripeNeffs.size();

    if (updated || inTemperature.changed || inWavelength.changed || polarization != old_polarization) { // We need to update something

        old_polarization = polarization;

        k0 = 2e3*M_PI / inWavelength();
        double w = real(inWavelength());

        logger(LOG_DEBUG, "Updating refractive indices cache");
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

        // Compute effective indices for all stripes
        // TODO: start form the stripe with highest refractive index and use effective index of adjacent stripe to find the new one

        for (size_t i = xbegin; i != nrCache.size(); ++i) {

            logger(LOG_DETAIL, "Computing effective index for vertical stripe no %1% (polarization %2%)", i, (polarization==TE)?"TE":"TM");
            std::stringstream nrs; for (auto nr: nrCache[i]) nrs << ", " << str(nr);
            logger(LOG_DEBUG, "nR[%1%] = [%2% ]", i, nrs.str().substr(1));

            dcomplex same_val = nrCache[i].front();
            bool all_the_same = true;
            for (auto n: nrCache[i]) {
                if (n != same_val) { all_the_same = false; break; }
            }
            if (all_the_same) {
                stripeNeffs[i] = same_val;
            } else {
                RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i]);}, log_stripe, 1e-4, 1e-6, 1e-5, 1.0, maxiterations);
                dcomplex maxn = *std::max_element(nrCache[i].begin(), nrCache[i].end(), [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
                stripeNeffs[i] = rootdigger.getSolution(0.999999*maxn);
                // dcomplex minn = *std::min_element(nrCache[i].begin(), nrCache[i].end(), [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
                // auto map = rootdigger.findMap(0.999999*maxn, 1.000001*minn, initial_stripe_neff_map, 0);
                // stripeNeffs[i] = rootdigger.getSolution(map[0]);
            }
        }
        if (xbegin == 1) {
            nrCache[0] = nrCache[1];
            stripeNeffs[0] = stripeNeffs[1];
        }
    }
    std::stringstream nrs; for (size_t i = xbegin; i < N; ++i) nrs << ", " << str(stripeNeffs[i]);
    logger(LOG_DEBUG, "stripes neffs = [%1% ]", nrs.str().substr(1));
}



/********* Here are the computations *********/

/* It would probably be better to use S-matrix method, but for simplicity we use T-matrix */

using namespace Eigen;

Eigen::Matrix2cd EffectiveIndex2dModule::getMatrix1(const plask::dcomplex& neff, const std::vector<dcomplex>& NR)
{
    size_t N = NR.size();

    std::vector<dcomplex> beta(N);
    for (size_t i = 0; i < N; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - neff*neff);
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TM)? NR[i+1]/NR[i] : 1.;
        dcomplex n = 0.5 * beta[i]/beta[i+1] * f*f;
        Matrix2cd M; M << (0.5+n), (0.5-n),
                        (0.5-n), (0.5+n);
        return M;
    };

    Matrix2cd T = fresnel(0);

    for (size_t i = 1; i < N-1; ++i) {
        double d = mesh->c1[i] - mesh->c1[i-1];
        dcomplex phas = exp(-I * beta[i] * d);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        T = P * T;
        T = fresnel(i) * T;
    }

    return T;
}

dcomplex EffectiveIndex2dModule::detS1(const plask::dcomplex& x, const std::vector<dcomplex>& NR)
{
    Matrix2cd T = getMatrix1(x, NR);
    // Fn = | T00 T01 | F0
    // Bn = | T10 T11 | B0
    return T(1,1);          // F0 = 0   Bn = 0
}


Matrix2cd EffectiveIndex2dModule::getMatrix(dcomplex neff)
{
    // Adjust for mirror losses
    neff = dcomplex(real(neff), imag(neff)-getMirrorLosses(neff));

    size_t N = stripeNeffs.size();

    std::vector<dcomplex> beta(N);
    for (size_t i = xbegin; i < N; ++i) {
        beta[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
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


dcomplex EffectiveIndex2dModule::detS(const dcomplex& x)
{
    Matrix2cd T = getMatrix(x);
    // Rn = | T00 T01 | R0
    // Ln = | T10 T11 | L0
    if (symmetry == SYMMETRY_POSITIVE) return T(1,0) + T(1,1);      // R0 = L0   Ln = 0
    else if (symmetry == SYMMETRY_NEGATIVE) return T(1,0) - T(1,1); // R0 = -L0  Ln = 0
    else return T(1,1);                                             // R0 = 0    Ln = 0
}



const DataVector<double> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod)
{
    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    dcomplex neff = outNeff();

    size_t Nx = mesh->tran().size()+1;
    std::vector<dcomplex> betax(Nx);
    for (size_t i = 0; i < Nx; ++i) {
        betax[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
    }
    auto fresnelX = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TE)? stripeNeffs[i]/stripeNeffs[i+1] :  1.;
        dcomplex n = 0.5 * betax[i+1]/betax[i] * f*f;
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };
    if (!have_fields) {
        fieldX.resize(Nx);
        fieldX[Nx-1] << 1., 0;
        for (size_t i = Nx-2; i >= 0; --i) {
            fieldX[i].noalias() = fresnelX(i) * fieldX[i+1];
            double d = (symmetry == NO_SYMMETRY)? mesh->tran()[i] - mesh->tran()[max(int(i)-1, 0)] :
                       (i == 0)? mesh->tran()[0] : mesh->tran()[i] - mesh->tran()[i-1];
            dcomplex phas = exp(- I * betax[i] * d);
            DiagonalMatrix<dcomplex, 2> P;
            P.diagonal() << 1./phas, phas;  // we propagate backward
            fieldX[i] = P * fieldX[i];
        }
    }


    size_t Ny = mesh->up().size()+1;
    // TODO better choice for field. Maybe some averaging?
    size_t mid_x = (symmetry == NO_SYMMETRY)? Nx/2. : 0;
    std::vector<dcomplex> betay(Ny);
    for (size_t i = 0; i < Ny; ++i) {
        betay[i] = k0 * sqrt(nrCache[mid_x][i]*nrCache[mid_x][i] - neff*neff);
    }
    auto fresnelY = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TM)? nrCache[mid_x][i]/nrCache[mid_x][i+1] :  1.;
        dcomplex n = 0.5 * betay[i+1]/betay[i] * f*f;
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };
    if (!have_fields) {
        fieldY.resize(Ny);
        fieldY[Ny-1] << 1., 0;
        for (size_t i = Ny-2; i >= 0; --i) {
            fieldY[i].noalias() = fresnelY(i) * fieldY[i+1];
            double d = mesh->up()[i] - mesh->up()[max(int(i)-1, 0)];
            dcomplex phas = exp(- I * betay[i] * d);
            DiagonalMatrix<dcomplex, 2> P;
            P.diagonal() << 1./phas, phas;  // we propagate backward
            fieldY[i] = P * fieldY[i];
        }
    }

    DataVector<double> results(dst_mesh.size());
    size_t idx = 0;

    for (auto point: dst_mesh) {
        double x = point.tran;

        bool negate = false;
        if (x < 0. && symmetry != NO_SYMMETRY) {
            x = -x;
            if (symmetry == SYMMETRY_NEGATIVE) negate = true;
        }

        size_t ix = mesh->tran().findIndex(x);
        x -= mesh->tran()[max(int(ix)-1, 0)];
        dcomplex phasx = exp(- I * betax[ix] * x);
        dcomplex val = fieldX[ix][0] * phasx + fieldX[ix][1] / phasx;
        if (negate) val = - val;


        double y = point.up;
        size_t iy = mesh->tran().findIndex(y);
        y -= mesh->up()[max(int(iy)-1, 0)];
        dcomplex phasy = exp(- I * betay[iy] * y);
        val *= fieldY[iy][0] * phasy + fieldY[iy][1] / phasy;

        results[idx++] = real(val*conj(val));
    }

    return results;
}




}}} // namespace plask::modules::eim
