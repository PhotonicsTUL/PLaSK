#include "eim.h"

namespace plask { namespace modules { namespace eim {

EffectiveIndex2dModule::EffectiveIndex2dModule() :
    symmetry(NO_SYMMETRY),
    tolx(1.0e-07),                                          // absolute tolerance on the argument
    tolf_min(1.0e-10),                                      // sufficient tolerance on the function value
    tolf_max(1.0e-8),                                       // required tolerance on the function value
    maxstep(0.1),                                           // maximum step in one iteration
    maxiterations(500),                                     // maximum number of iterations
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

    // Create middle-points mesh
    middle_points = RectilinearMesh2d(mesh->getMidpointsMesh());
    middle_points.c0.addPoint(mesh->c0[0] - outer_distance);
    middle_points.c0.addPoint(mesh->c0[mesh->c0.size()-1] + outer_distance);
    middle_points.c1.addPoint(mesh->c1[0] - outer_distance);
    middle_points.c1.addPoint(mesh->c1[mesh->c1.size()-1] + outer_distance);

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(mesh->tran().size()+1, std::vector<dcomplex>(mesh->up().size()+1));
    stripeNeffs.resize(mesh->tran().size()+1);
}


void EffectiveIndex2dModule::onInvalidate()
{
    outNeff.invalidate();
}



void EffectiveIndex2dModule::stageOne()
{
    bool updated = initCalculation();

    size_t xbegin = 0;

    // Some additional checks
    if (symmetry == SYMMETRY_POSITIVE || symmetry == SYMMETRY_NEGATIVE) {
        if (geometry->isSymmetric(CalculationSpace::DIRECTION_TRAN)) {
            if (updated) // Make sure we have only positive points
                for (auto x: mesh->c0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
            xbegin = 1;
        } else {
            log(LOG_WARNING, "Symmetry reset to NO_SYMMETRY for non-symmetric geometry.");
            symmetry = NO_SYMMETRY;
        }
    }

    size_t xsize = middle_points.c0.size();
    size_t ysize = middle_points.c1.size();
    size_t txmax = mesh->c0.size() - 1;
    size_t tymax = mesh->c1.size() - 1;

    if (updated || inTemperature.changed || inWavelength.changed) {
        // Either temperature, structure, or wavelength changed, so we need to get refractive indices
        log(LOG_DEBUG, "Updating refractive indices cache");
        k0 = 2*M_PI / inWavelength();;
        double w = real(k0);
        auto temp = inTemperature(*mesh);
        for (size_t i = xbegin; i != xsize; ++i) {
            size_t tx0 = (i > 0)? i - 1 : 0;
            size_t tx1 = (i < txmax)? i : txmax;
            for (size_t j = 0; j != ysize; ++j) {
                size_t ty0 = (j > 0)? j - 1 : 0;
                size_t ty1 = (j < tymax)? j : txmax;
                double T = 0.25 * ( temp[mesh->index(tx0, ty0)] + temp[mesh->index(tx0, ty1)] +
                                    temp[mesh->index(tx1, ty0)] + temp[mesh->index(tx1, ty1)] );
                nrCache[i][j] = geometry->getMaterial(middle_points(i,j))->Nr(w, T);
            }
        }
    }

    size_t N = stripeNeffs.size();

    // Compute effective indices for all stripes
    // TODO: start form the stripe with highest refractive index and use effective index of adjacent stripe to find the new one
    log(LOG_INFO, "Computing effective index for each vertical stripe");
    for (size_t i = xbegin; i != nrCache.size(); ++i) {
        auto maxn = *std::max_element(nrCache[i].begin(), nrCache[i].end(), [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
        log(LOG_DEBUG, "Computing effective index for vertical stripe no %1% (x=%2%µm)", i, middle_points.c0[i]);
        stripeNeffs[i] = RootDigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i]);} ).getSolution(maxn);
    }
}



/********* Here are the computations *********/

/* It would probably be better to use S-matrix method, but for simplicity we use T-matrix */

using namespace Eigen;

Eigen::Matrix2cd EffectiveIndex2dModule::getMatrix1(const plask::dcomplex& neff, const std::vector<dcomplex>& NR)
{
    size_t N = NR.size();

    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex n = 0.5 * NR[i] / NR[i+1];
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };

    Matrix2cd T = fresnel(0);

    for (size_t i = 1; i < N-1; ++i) {
        dcomplex n = sqrt(NR[i]*NR[i] - neff*neff);
        if (real(n) < 0) n = -n; // Is this necessary? Just to be on the safe side?
        double d = mesh->c1[i] - mesh->c1[i-1];
        dcomplex phas = exp(-I * n * d * k0);
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

    Matrix2cd S; // Scattering matrix
    S <<     1. / T(0,0)  ,  - T(0,1) / T(0,0),
         T(1,0) / T(0,0)  ,  - T(0,1) / T(0,0) * T(1,0) + T(1,1);
std::cout << T << "\n" << S << "\n";
    return S.determinant();
}


Matrix2cd EffectiveIndex2dModule::getMatrix(const dcomplex& neff)
{
    size_t N = stripeNeffs.size();
    size_t xbegin = (symmetry == NO_SYMMETRY)? 0 : 1;


    auto fresnel = [&](size_t i) -> Matrix2cd {
        dcomplex n = 0.5 * stripeNeffs[i] / stripeNeffs[i+1];
        Matrix2cd M; M << (0.5+n), (0.5-n),
                          (0.5-n), (0.5+n);
        return M;
    };

    Matrix2cd T = fresnel(0);

    if (xbegin) { // we have symmetry, so begin of transfer matrix is at the axis
        dcomplex n = sqrt(stripeNeffs[1]*stripeNeffs[1] - neff*neff);
        if (real(n) < 0) n = -n; // Is this necessary? Just to be on the safe side?
        dcomplex phas = exp(-I * n * mesh->c0[1] * k0);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        T = P * T;
    }

    for (size_t i = xbegin+1; i < N-1; ++i) {
        dcomplex n = sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
        if (real(n) < 0) n = -n; // Is this necessary? Just to be on the safe side?
        double d = mesh->c0[i] - mesh->c0[i-1];
        dcomplex phas = exp(-I * n * d * k0);
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

    if (symmetry == NO_SYMMETRY) {
        Matrix2cd S; // Scattering matrix
        S <<     1. / T(0,0)  ,  - T(0,1) / T(0,0),
             T(1,0) / T(0,0)  ,  - T(0,1) / T(0,0) * T(1,0) + T(1,1);
        return S.determinant();
    } // else TODO
}



const DataVector<double> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{
    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    //TODO
    DataVector<double> data(4);
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;
    return data;
}




}}} // namespace plask::modules::eim
