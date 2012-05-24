#include "eim.h"

using plask::dcomplex;

namespace plask { namespace modules { namespace eim {

EffectiveIndex2dModule::EffectiveIndex2dModule() :
    have_fields(false),
    old_polarization(TE),
    polarization(TE),
    symmetry(NO_SYMMETRY),
    outer_distance(0.1),
    tolx(1.0e-07),
    tolf_min(1.0e-10),
    tolf_max(1.0e-8),
    maxstep(0.1),
    maxiterations(500),
    log_value(dataLog<dcomplex, double>(getId(), "neff", "char_val")),
    outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
    inTemperature = 300.;
}


dcomplex EffectiveIndex2dModule::computeMode(dcomplex neff)
{
    log(LOG_INFO, "Searching for the mode starting from n_eff = %1%", str(neff));
    stageOne();
    dcomplex result = RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}).getSolution(neff);
    outNeff = result;
    have_fields = false;
    return result;
}



std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex neff1, dcomplex neff2, unsigned steps, unsigned nummodes)
{
    log(LOG_INFO, "Searching for the modes for n_eff between %1% and %2%", str(neff1), str(neff2));
    stageOne();
    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}).searchSolutions(neff1, neff2, steps, 0, nummodes);
}



std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps)
{
    log(LOG_INFO, "Searching for the approximate modes for n_eff between %1% and %2%", str(neff1), str(neff2));
    stageOne();

    double rdneff = real(neff2 - neff1);
    double rneff1 = real(neff1);
    double steps1 = steps + 1;
    std::vector<double> rpoints(steps+1);
    for (unsigned i = 0; i <= steps; ++i) {
        rpoints[i] = rneff1 + rdneff * i / steps1;
    }

    std::vector<double> ipoints(1, imag(neff1));
    if (imag(neff2) != imag(neff1)) ipoints.push_back(imag(neff2));

    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}).findMap(rpoints, ipoints);
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
            log(LOG_WARNING, "Symmetry reset to NO_SYMMETRY for non-symmetric geometry.");
            symmetry = NO_SYMMETRY;
        }
    }

    size_t xsize = mesh->c0.size() + 1;
    size_t ysize = mesh->c1.size() + 1;

    size_t N = stripeNeffs.size();

    if (updated || inTemperature.changed || inWavelength.changed || polarization != old_polarization) { // We need to update something

        old_polarization = polarization;

        k0 = 2e3*M_PI / inWavelength();
        double w = real(inWavelength());

        log(LOG_DEBUG, "Updating refractive indices cache");
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

            log(LOG_DETAIL, "Computing effective index for vertical stripe no %1%", i);
            std::stringstream nrs; for (auto nr: nrCache[i]) nrs << ", " << str(nr);
            log(LOG_DEBUG, "nR[%1%] = [%2% ]", i, nrs.str().substr(1));

            dcomplex same_val = nrCache[i].front();
            bool all_the_same = true;
            for (auto n: nrCache[i]) {
                if (n != same_val) { all_the_same = false; break; }
            }
            if (all_the_same) {
                stripeNeffs[i] = same_val;
            } else {
                auto maxn = *std::max_element(nrCache[i].begin(), nrCache[i].end(), [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
                stripeNeffs[i] = RootDigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i]);} ).getSolution(0.9999*maxn);
            }
        }
    }
    std::stringstream nrs; for (int i = xbegin; i < N; ++i) nrs << ", " << str(stripeNeffs[i]);
    log(LOG_DEBUG, "stripes neffs = [%1% ]", nrs.str().substr(1));
}



/********* Here are the computations *********/

/* It would probably be better to use S-matrix method, but for simplicity we use T-matrix */

// TODO TM polarization

using namespace Eigen;

Eigen::Matrix2cd EffectiveIndex2dModule::getMatrix1(const plask::dcomplex& neff, const std::vector<dcomplex>& NR)
{
    size_t N = NR.size();

    std::vector<dcomplex> beta(N);
    for (int i = 0; i < N; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - neff*neff);
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TE)? 1. : NR[i+1]/NR[i];
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


Matrix2cd EffectiveIndex2dModule::getMatrix(const dcomplex& neff)
{
    size_t N = stripeNeffs.size();

    std::vector<dcomplex> beta(N);
    for (int i = xbegin; i < N; ++i) {
        beta[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
    }

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex f =  (polarization==TM)? 1. : stripeNeffs[i+1]/stripeNeffs[i];
        dcomplex n = 0.5 * beta[i]/beta[i+1] * f*f;
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };


    Matrix2cd T = fresnel(xbegin);

    if (symmetry != NO_SYMMETRY) { // we have symmetry, so begin of transfer matrix is at the axis
        dcomplex phas = exp(-I * beta[xbegin] * mesh->c0[xbegin]);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        T = T * P;
    }

    for (size_t i = xbegin+1; i < N-1; ++i) {
        dcomplex n = sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
        double d = mesh->c0[i] - mesh->c0[i-1];
        dcomplex phas = exp(-I * beta[i] * d);
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



const DataVector<double> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{
    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    if (!have_fields) {

        size_t Nx = mesh->tran().size()+1;
        size_t Ny = mesh->up().size()+1;
        size_t Nx2 = Nx/2.;
        size_t Ny2 = Ny/2.;

        fieldX.resize(Nx);

        Matrix2cd T = getMatrix(outNeff());
        // R0 = | iT00 iT01 | Rn   Rn = 1
        // L0 = | iT10 iT11 | Ln   Ln = 0
        fieldX[Nx-1] << 1., 0;
        fieldX[xbegin].noalias() = T.inverse() * fieldX[Nx-1];

        for (size_t ix = xbegin; ix <= Nx2; ++ix) {
        }
        for (size_t ix = Nx-1; ix > Nx2; --ix) {
        }


//         fieldsY.resize(Nx); for (auto vec: fieldsY) vec.resize(Ny);
//         for (size_t ix = xbegin; ix < Nx; ++ix) {
//             fieldF[ix][0] = 0.; fieldB[ix][Ny-1] = 0.;                      // Fn = | T00 T01 | F0
//             fieldB[ix][0] = 1.; fieldF[ix][Ny-1] = T(0,1) * fieldB[ix][0];  // Bn = | T10 T11 | B0
//             size_t Ny2 = Ny/2.;
//             for (size_t iy = 0; iy < Ny2; ++iy) {
//             }
//             for (size_t iy = Ny-1; iy >= Ny2; --iy) {
//             }
//
//         }
    }


    //TODO
    DataVector<double> data(4);
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;
    return data;
}




}}} // namespace plask::modules::eim
