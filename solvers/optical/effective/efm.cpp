#include "efm.h"
#include "amos/amos.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

EffectiveFrequencyCylSolver::EffectiveFrequencyCylSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCylindrical, RectilinearMesh2D>(name),
    log_value(dataLog<dcomplex, dcomplex>("freq", "v", "det")),
    have_fields(false),
    l(0),
    k0(NAN),
    outer_distance(0.1),
    outIntensity(this, &EffectiveFrequencyCylSolver::getLightIntenisty) {
    inTemperature = 300.;
    inGain = NAN;
    root.tolx = 1.0e-9;
    root.tolf_min = 1.0e-12;
    root.tolf_max = 1.0e-8;
    root.maxstep = 0.1;
    root.maxiterations = 500;
    striperoot.tolx = 1.0e-9;
    striperoot.tolf_min = 1.0e-12;
    striperoot.tolf_max = 1.0e-8;
    striperoot.maxstep = 0.5;
    striperoot.maxiterations = 500;
}


void EffectiveFrequencyCylSolver::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
            l = reader.getAttribute<unsigned short>("l", l);
            auto alam0 = reader.getAttribute<unsigned short>("lam0");
            auto ak0 = reader.getAttribute<unsigned short>("k0");
            if (alam0) {
                if (ak0) throw XMLConflictingAttributesException(reader, "k0", "lam0");
                k0 = 2e3*M_PI / *alam0;
            } else if (ak0) k0 = *ak0;
            reader.requireTagEnd();
        } else if (param == "root") {
            root.tolx = reader.getAttribute<double>("tolx", root.tolx);
            root.tolf_min = reader.getAttribute<double>("tolf_min", root.tolf_min);
            root.tolf_max = reader.getAttribute<double>("tolf_max", root.tolf_max);
            root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
            root.maxiterations = reader.getAttribute<int>("maxiterations", root.maxstep);
            reader.requireTagEnd();
        } else if (param == "striperoot") {
            striperoot.tolx = reader.getAttribute<double>("tolx", striperoot.tolx);
            striperoot.tolf_min = reader.getAttribute<double>("tolf_min", striperoot.tolf_min);
            striperoot.tolf_max = reader.getAttribute<double>("tolf_max", striperoot.tolf_max);
            striperoot.maxstep = reader.getAttribute<double>("maxstep", striperoot.maxstep);
            striperoot.maxiterations = reader.getAttribute<int>("maxiterations", striperoot.maxiterations);
            reader.requireTagEnd();
        } else if (param == "outer") {
            outer_distance = reader.requireAttribute<double>("distance");
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <mode>, <root>, <striperoot>, or <outer>");
    }
}

dcomplex EffectiveFrequencyCylSolver::computeMode(dcomplex lambda)
{
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = %1%", str(lambda));
    if (isnan(k0.real())) k0 = 2e3*M_PI / lambda;
    stageOne();
    v = RootDigger(*this, [this](const dcomplex& v){return this->detS(v);}, log_value, root).getSolution(0.);
    dcomplex k = k0 / (1. - v/2.); // get wavelength back from frequency parameter
    dcomplex lam = 2e3*M_PI / k;
    outWavelength = real(lam);
    outExtinction = phys::c * imag(k);
    outWavelength.fireChanged();
    outExtinction.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return lam;
}



std::vector<dcomplex> EffectiveFrequencyCylSolver::findModes(dcomplex lambda1, dcomplex lambda2, unsigned steps, unsigned nummodes)
{
    writelog(LOG_INFO, "Searching for the modes for wavelength between %1% and %2%", str(lambda1), str(lambda2));
    if (isnan(k0.real())) k0 = 4e3*M_PI / (lambda1 + lambda2);
    stageOne();
    dcomplex v1 = 2. * (k0 - (2e3*M_PI/lambda1)) / k0;
    dcomplex v2 = 2. * (k0 - (2e3*M_PI/lambda2)) / k0;
    auto results = RootDigger(*this, [this](const dcomplex& v){return this->detS(v);}, log_value, root)
                       .searchSolutions(v1, v2, steps, 0, nummodes);
    for (auto& res: results) res = 2e3*M_PI / k0 / (1. - res/2.); // get wavelengths back from frequency parameter
    return results;
}

std::vector<dcomplex> EffectiveFrequencyCylSolver::findModesMap(dcomplex lambda1, dcomplex lambda2, unsigned steps)
{
    writelog(LOG_INFO, "Searching for the approximate modes for wavelength between %1% and %2%", str(lambda1), str(lambda2));
    if (isnan(k0.real())) k0 = 4e3*M_PI / (lambda1 + lambda2);
    stageOne();
    dcomplex v1 = 2. * (k0 - (2e3*M_PI/lambda1)) / k0;
    dcomplex v2 = 2. * (k0 - (2e3*M_PI/lambda2)) / k0;
    auto results =  RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, root)
                        .findMap(v1, v2, steps, 0);
    for (auto& res: results) res = 2e3*M_PI / k0 / (1. - res/2.); // get wavelengths back from frequency parameter
    return results;
}

void EffectiveFrequencyCylSolver::setMode(dcomplex clambda)
{
    if (isnan(k0.real())) k0 = 2e3*M_PI / clambda;
    if (!initialized) {
        writelog(LOG_WARNING, "Solver invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    v =  2. - 4e3*M_PI / clambda / k0;
    double det = abs(detS(v));
    if (det > root.tolf_max) throw BadInput(getId(), "Provided wavelength does not correspond to any mode (det = %1%)", det);
    writelog(LOG_INFO, "Setting current mode to %1%", str(clambda));
    outWavelength = real(clambda);
    outExtinction = phys::c * imag(2e3*M_PI/clambda);
    outWavelength.fireChanged();
    outExtinction.fireChanged();
    outIntensity.fireChanged();
}


void EffectiveFrequencyCylSolver::onInitialize()
{
    if (!geometry) throw NoGeometryException(getId());

    // Set default mesh
    if (!mesh) setSimpleMesh();

    // Assign space for refractive indices cache and stripe effective indices
    size_t rsize = mesh->axis0.size();

    if (geometry->getBorder(Geometry::DIRECTION_TRAN, true).type() == border::Strategy::EXTEND &&
        abs(mesh->axis0[mesh->axis0.size()-1] - geometry->getChild()->getBoundingBox().upper[0]) < SMALL)
        --rsize;

    nrCache.assign(rsize, std::vector<dcomplex>(mesh->axis1.size()+1));
    ngCache.assign(rsize, std::vector<dcomplex>(mesh->axis1.size()+1));
    veffs.resize(rsize);
    nng.resize(rsize);
}


void EffectiveFrequencyCylSolver::onInvalidate()
{
    outWavelength.invalidate();
    outExtinction.invalidate();
    have_fields = false;
    outWavelength.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/

static const int mh = 2; // Hankel function type (1 or 2)

using namespace Eigen;


void EffectiveFrequencyCylSolver::stageOne()
{
    bool fresh = !initCalculation();

    // Some additional checks
    for (auto x: mesh->axis0) {
        if (x < 0.) throw BadMesh(getId(), "for cylindrical geometry no radial points can be negative");
    }
    if (abs(mesh->axis0[0]) > SMALL) throw BadMesh(getId(), "radial mesh must start from zero");

    size_t rsize = veffs.size();
    size_t zsize = mesh->axis1.size() + 1;

    if (fresh || inTemperature.changed || inGain.changed || l != old_l || k0 != old_k0) { // we need to update something

        old_l = l;
        old_k0 = k0;

        double lam = real(2e3*M_PI / k0);

        writelog(LOG_DEBUG, "Updating refractive indices cache");
        auto temp = inTemperature(*mesh);

        double h = lam * 1e6*SMALL;
        double lam1 = lam - h, lam2 = lam + h;
        double ih2 = 0.5 / h;

        auto midmesh = mesh->getMidpointsMesh();
        auto gain = inGain(midmesh, lam1);
        auto gain_slope = inGain(midmesh, lam2).claim();
        {
            auto g1 = gain_slope.begin();
            auto g2 = gain.begin();
            for (; g1 != gain_slope.end(); ++g1, ++g2) *g1 = (*g2 - *g1) * ih2;
        }
        gain = inGain(midmesh, lam);

        for (size_t ix = 0; ix != rsize; ++ix) {
            size_t tx1;
            double x0, x1;
            x0 = mesh->axis0[ix];
            if (ix < mesh->axis0.size()-1) { tx1 = ix+1; x1 = mesh->axis0[tx1]; } else { tx1 = mesh->axis0.size()-1; x1 = mesh->axis0[tx1] + 2.*outer_distance; }
            for (size_t iy = 0; iy < zsize; ++iy) {
                size_t ty0, ty1;
                double y0, y1;
                if (iy > 0) { ty0 = iy-1; y0 = mesh->axis1[ty0]; } else { ty0 = 0; y0 = mesh->axis1[ty0] - 2.*outer_distance; }
                if (iy < zsize-1) { ty1 = iy; y1 = mesh->axis1[ty1]; } else { ty1 = zsize-2; y1 = mesh->axis1[ty1] + 2.*outer_distance; }

                double T = 0.25 * ( temp[mesh->index(ix,ty0)] + temp[mesh->index(ix,ty1)] +
                                    temp[mesh->index(tx1,ty0)] + temp[mesh->index(tx1,ty1)] );

                auto material = geometry->getMaterial(0.25 * (vec(x0,y0) + vec(x0,y1) + vec(x1,y0) + vec(x1,y1)));

                // Nr = nr + i/(4π) λ g
                // Ng = Nr - λ dN/dλ = Nr - λ dn/dλ - i/(4π) λ^2 dg/dλ
                nrCache[ix][iy] = material->Nr(lam, T);
                ngCache[ix][iy] = nrCache[ix][iy] - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) * ih2;
                double g = (ix == rsize-1 || iy == 0 || iy == zsize-1)? NAN : gain[midmesh->index(ix, iy-1)];
                double gs = (ix == rsize-1 || iy == 0 || iy == zsize-1)? NAN : gain_slope[midmesh->index(ix, iy-1)];
                ngCache[ix][iy] -= dcomplex(0., (isnan(gs))? 0. : 7.95774715459e-09 * lam*lam * gs);
                nrCache[ix][iy] += dcomplex(0., isnan(g)? 0. : 7.95774715459e-09 * lam * g);

            }
        }

        // Compute effective frequencies for all stripes
        // #pragma omp parallel for // UNEFFICIENT
        for (size_t i = 0; i < nrCache.size(); ++i) {

            writelog(LOG_DETAIL, "Computing effective frequency for vertical stripe %1%", i);
#           ifndef NDEBUG
                std::stringstream nrgs; for (auto nr = nrCache[i].begin(), ng = ngCache[i].begin(); nr != nrCache[i].end(); ++nr, ++ng)
                    nrgs << ", (" << str(*nr) << ")/(" << str(*ng) << ")";
                writelog(LOG_DEBUG, "nR/nG[%1%] = [%2% ]", i, nrgs.str().substr(1));
#           endif

            dcomplex same_nr = nrCache[i].front();
            dcomplex same_ng = ngCache[i].front();
            bool all_the_same = true;
            for (auto nr = nrCache[i].begin(), ng = ngCache[i].begin(); nr != nrCache[i].end(); ++nr, ++ng)
                if (*nr != same_nr || *ng != same_ng) { all_the_same = false; break; }
            if (all_the_same) {
                veffs[i] = 1.; // TODO make sure this is so!
                nng[i] = same_nr * same_ng;
            } else {
                Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[%1%]", i), "veff", "det");
                RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i],ngCache[i]);}, log_stripe, striperoot);
                veffs[i] = rootdigger.getSolution(1e-3);
                computeStripeNNg(i);
            }
        }

#       ifndef NDEBUG
            std::stringstream strv; for (size_t i = 0; i < veffs.size(); ++i) strv << ", " << str(veffs[i]);
            writelog(LOG_DEBUG, "stripes veffs = [%1% ]", strv.str().substr(1));
            std::stringstream strn; for (size_t i = 0; i < nng.size(); ++i) strn << ", " << str(nng[i]);
            writelog(LOG_DEBUG, "stripes <nng> = [%1% ]", strn.str().substr(1));
#       endif
    }
}


dcomplex EffectiveFrequencyCylSolver::detS1(const dcomplex& v, const std::vector<dcomplex>& NR, const std::vector<dcomplex>& NG)
{
    size_t N = NR.size();

    std::vector<dcomplex> beta(N);
    for (size_t i = 0; i < N; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - v * NR[i]*NG[i]);
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

void EffectiveFrequencyCylSolver::computeStripeNNg(size_t stripe)
{
    size_t N = nrCache[stripe].size();
    dcomplex veff = veffs[stripe];

    nng[stripe] = 0.;
    dcomplex sum = 0.;

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
        // double f = real(E[0]*conj(E[0])) + real(E[1]*conj(E[1]));
        dcomplex f = E[0]*E[0] + E[1]*E[1]; //TODO
        nng[stripe] += f * nrCache[stripe][i] * ngCache[stripe][i];
        sum += f;
        dcomplex phas = exp(-I * beta[i] * d);
        DiagonalMatrix<dcomplex, 2> P;
        P.diagonal() << phas, 1./phas;
        E = P * E;
        E = fresnel(i) * E;
    }

    nng[stripe] /= sum;
}


Matrix2cd EffectiveFrequencyCylSolver::getMatrix(dcomplex v, size_t i)
{
    double r = mesh->axis0[i];

    dcomplex x1 = r * k0 * sqrt(nng[i-1] * (veffs[i-1]-v));
    if (real(x1) < 0.) x1 = -x1;

    dcomplex x2 = r * k0 * sqrt(nng[i] * (veffs[i]-v));
    if (real(x2) < 0.) x2 = -x2;

    Matrix2cd A, B;

    // Compute Bessel functions and their derivatives
    dcomplex J1[2], H1[2];
    dcomplex J2[2], H2[2];
    double Jr[2], Ji[2], Hr[2], Hi[2];
    int nz, ierr;

    F77_GLOBAL(zbesj,ZBESJ)(x1.real(), x1.imag(), l, 1, 2, Jr, Ji, nz, ierr);
    if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", l, str(x1));
    F77_GLOBAL(zbesh,ZBESH)(x1.real(), x1.imag(), l, 1, mh, 2, Hr, Hi, nz, ierr);
    if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", l, str(x1));
    for (int i = 0; i < 2; ++i) { J1[i] = dcomplex(Jr[i], Ji[i]); H1[i] = dcomplex(Hr[i], Hi[i]); }

    F77_GLOBAL(zbesj,ZBESJ)(x2.real(), x2.imag(), l, 1, 2, Jr, Ji, nz, ierr);
    if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", l, str(x2));
    F77_GLOBAL(zbesh,ZBESH)(x2.real(), x2.imag(), l, 1, mh, 2, Hr, Hi, nz, ierr);
    if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", l, str(x2));
    for (int i = 0; i < 2; ++i) { J2[i] = dcomplex(Jr[i], Ji[i]); H2[i] = dcomplex(Hr[i], Hi[i]); }


    A <<   J1[0],                 H1[0],
         l*J1[0] - x1*J1[1],    l*H1[0] - x1*H1[1];

    B <<   J2[0],                 H2[0],
         l*J2[0] - x2*J2[1],    l*H2[0] - x2*H2[1];

    return A.inverse() * B;
}


dcomplex EffectiveFrequencyCylSolver::detS(const dcomplex& v)
{
    Vector2cd E;

    // In the outermost layer, there is only an outgoing wave, so the solution is only the Hankel function
    E << 0., 1.;

    for (size_t i = veffs.size()-1; i > 0; --i) {
        E = getMatrix(v, i) * E;
    }

    // In the innermost area there must not be any infinity, so H = 0.
    //return E[1] / E[0]; // to stress the difference between J and H
    return E[1];
}



plask::DataVector<const double> EffectiveFrequencyCylSolver::getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod)
{
    if (!outWavelength.hasValue() || k0 != old_k0 || l != old_l) throw NoValue(OpticalIntensity::NAME);

    if (!have_fields) {
        fieldR.resize(veffs.size());
        fieldR[veffs.size()-1] << 0., 1.;

        writelog(LOG_INFO, "Computing field distribution for wavelength = %1%", str(outWavelength()));

        // Compute horizontal part
        for (size_t i = veffs.size()-1; i > 0; --i) {
            fieldR[i-1].noalias() = getMatrix(v, i) * fieldR[i];
        }

#       ifndef NDEBUG
            std::stringstream strf; for (size_t i = 0; i < fieldR.size(); ++i) strf << ", (" << str(fieldR[i][0]) << ")/(" << str(fieldR[i][1]) << ")";
            writelog(LOG_DEBUG, "E=aY+bH: a/b = [%1% ]", strf.str().substr(1));
#       endif

        size_t stripe = 0;
        // Look for the innermost stripe with not constant refractive index
        bool all_the_same = true;
        while (all_the_same) {
            dcomplex same_nr = nrCache[stripe].front();
            dcomplex same_ng = ngCache[stripe].front();
            for (auto nr = nrCache[stripe].begin(), ng = ngCache[stripe].begin(); nr != nrCache[stripe].end(); ++nr, ++ng)
                if (*nr != same_nr || *ng != same_ng) { all_the_same = false; break; }
            if (all_the_same) ++ stripe;
        }
        writelog(LOG_DETAIL, "Vertical field distribution taken from stripe %1%", stripe);

        // Compute vertical part
        std::vector<dcomplex>& NR = nrCache[stripe];
        std::vector<dcomplex>& NG = ngCache[stripe];
        dcomplex veff = veffs[stripe];
        betaz.resize(NR.size());
        for (size_t i = 0; i < NR.size(); ++i) {
            betaz[i] = k0 * sqrt(NR[i]*NR[i] - veff * NR[i]*NG[i]);
            if (real(betaz[i]) < 0.) betaz[i] = -betaz[i];  // TODO verify this condition; in general it should consider really outgoing waves
        }
        auto fresnel = [&](size_t i) -> Matrix2cd {
            dcomplex n = 0.5 * betaz[i]/betaz[i+1];
            Matrix2cd M; M << (0.5+n), (0.5-n),
                              (0.5-n), (0.5+n);
            return M;
        };
        fieldZ.resize(NR.size());
        fieldZ[0] << 0., 1.;
        fieldZ[1].noalias() = fresnel(0) * fieldZ[0];
        for (size_t i = 1; i < NR.size()-1; ++i) {
            double d = mesh->axis1[i] - mesh->axis1[i-1];
            dcomplex phas = exp(-I * betaz[i] * d);
            DiagonalMatrix<dcomplex, 2> P;
            P.diagonal() << phas, 1./phas;
            fieldZ[i+1].noalias() = P * fieldZ[i];
            fieldZ[i+1] = fresnel(i) * fieldZ[i+1];
        }
    }

    DataVector<double> results(dst_mesh.size());

    if (!getLightIntenisty_Efficient<RectilinearMesh2D>(dst_mesh, results) &&
        !getLightIntenisty_Efficient<RegularMesh2D>(dst_mesh, results)) {

        #pragma omp parallel for schedule(static,1024)
        for (size_t id = 0; id < dst_mesh.size(); ++id) {
            auto point = dst_mesh[id];
            double r = point.c0;
            double z = point.c1;
            if (r < 0) r = -r;

            double Jr, Ji, Hr, Hi;
            int nz, ierr;

            size_t ir = mesh->axis0.findIndex(r); if (ir > 0) --ir; if (ir >= veffs.size()) ir = veffs.size()-1;
            dcomplex x = r * k0 * sqrt(nng[ir-1] * (veffs[ir-1]-v));
            if (real(x) < 0.) x = -x;
            F77_GLOBAL(zbesj,ZBESJ)(x.real(), x.imag(), l, 1, 1, &Jr, &Ji, nz, ierr);
            if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", l, str(x));
            if (ir == 0) {
                Hr = Hi = 0.;
            } else {
                F77_GLOBAL(zbesh,ZBESH)(x.real(), x.imag(), l, 1, mh, 1, &Hr, &Hi, nz, ierr);
                if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", l, str(x));
            }
            dcomplex val = fieldR[ir][0] * dcomplex(Jr, Ji) + fieldR[ir][1] * dcomplex(Hr, Hi);

            size_t iy = mesh->axis1.findIndex(z);
            z -= mesh->axis1[max(int(iy)-1, 0)];
            dcomplex phasz = exp(- I * betaz[iy] * z);
            val *= fieldZ[iy][0] * phasz + fieldZ[iy][1] / phasz;

            results[id] = real(abs2(val));
        }

    }

    // Normalize results to make maximum value equal to one
    double factor = 1. / *std::max_element(results.begin(), results.end());
    for (double& val: results) val *= factor;

    return results;
}

template <typename MeshT>
bool EffectiveFrequencyCylSolver::getLightIntenisty_Efficient(const plask::MeshD<2>& dst_mesh, plask::DataVector<double>& results)
{
    if (dynamic_cast<const MeshT*>(&dst_mesh)) {

        const MeshT& rect_mesh = dynamic_cast<const MeshT&>(dst_mesh);

        std::vector<dcomplex> valr(rect_mesh.axis0.size());
        std::vector<dcomplex> valz(rect_mesh.axis1.size());

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (size_t idr = 0; idr < rect_mesh.tran().size(); ++idr) {
                double r = rect_mesh.axis0[idr];
                double Jr, Ji, Hr, Hi;
                int nz, ierr;
                if (r < 0.) r = -r;
                size_t ir = mesh->axis0.findIndex(r); if (ir > 0) --ir;  if (ir >= veffs.size()) ir = veffs.size()-1;
                dcomplex x = r * k0 * sqrt(nng[ir] * (veffs[ir]-v));
                if (real(x) < 0.) x = -x;
                F77_GLOBAL(zbesj,ZBESJ)(x.real(), x.imag(), l, 1, 1, &Jr, &Ji, nz, ierr);
                if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", l, str(x));
                if (ir == 0) {
                    Hr = Hi = 0.;
                } else {
                    F77_GLOBAL(zbesh,ZBESH)(x.real(), x.imag(), l, 1, mh, 1, &Hr, &Hi, nz, ierr);
                    if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", l, str(x));
                }
                valr[idr] = fieldR[ir][0] * dcomplex(Jr, Ji) + fieldR[ir][1] * dcomplex(Hr, Hi);
            }

            #pragma omp for
            for (size_t idz = 0; idz < rect_mesh.up().size(); ++idz) {
                double z = rect_mesh.axis1[idz];
                size_t iz = mesh->axis1.findIndex(z);
                z -= mesh->axis1[max(int(iz)-1, 0)];
                dcomplex phasz = exp(- I * betaz[iz] * z);
                valz[idz] = fieldZ[iz][0] * phasz + fieldZ[iz][1] / phasz;
            }

            if (rect_mesh.getIterationOrder() == MeshT::NORMAL_ORDER) {
                #pragma omp for
                for (size_t i1 = 0; i1 < rect_mesh.axis1.size(); ++i1) {
                    double* data = results.data() + i1 * rect_mesh.axis0.size();
                    for (size_t i0 = 0; i0 < rect_mesh.axis0.size(); ++i0) {
                        dcomplex f = valr[i0] * valz[i1];
                        data[i0] = abs2(f);
                    }
                }
            } else {
                #pragma omp for
                for (size_t i0 = 0; i0 < rect_mesh.axis0.size(); ++i0) {
                    double* data = results.data() + i0 * rect_mesh.axis1.size();
                    for (size_t i1 = 0; i1 < rect_mesh.axis1.size(); ++i1) {
                        dcomplex f = valr[i0] * valz[i1];
                        data[i1] = abs2(f);
                    }
                }
            }
        }

        return true;
    }

    return false;
}


}}} // namespace plask::solvers::effective
