#include <camos/camos.h>

#include "efm.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

EffectiveFrequencyCylSolver::EffectiveFrequencyCylSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCylindrical, RectilinearMesh2D>(name),
    log_value(dataLog<dcomplex, dcomplex>("freq", "v", "det")),
    have_fields(false),
    m(0),
    k0(NAN),
    outdist(0.1),
    outIntensity(this, &EffectiveFrequencyCylSolver::getLightIntenisty) {
    inTemperature = 300.;
    inGain = NAN;
    root.tolx = 1.0e-8;
    root.tolf_min = 1.0e-10;
    root.tolf_max = 1.0e-6;
    root.maxstep = 0.1;
    root.maxiter = 500;
    stripe_root.tolx = 1.0e-8;
    stripe_root.tolf_min = 1.0e-10;
    stripe_root.tolf_max = 1.0e-6;
    stripe_root.maxstep = 0.05;
    stripe_root.maxiter = 500;
}


void EffectiveFrequencyCylSolver::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
            m = reader.getAttribute<unsigned short>("m", m);
            auto alam0 = reader.getAttribute<double>("lam0");
            auto ak0 = reader.getAttribute<double>("k0");
            if (alam0) {
                if (ak0) throw XMLConflictingAttributesException(reader, "k0", "lam0");
                k0 = 2e3*M_PI / *alam0;
            } else if (ak0) k0 = *ak0;
            reader.requireTagEnd();
        } else if (param == "root") {
            root.tolx = reader.getAttribute<double>("tolx", root.tolx);
            root.tolf_min = reader.getAttribute<double>("tolf-min", root.tolf_min);
            root.tolf_max = reader.getAttribute<double>("tolf-max", root.tolf_max);
            root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
            root.maxiter = reader.getAttribute<int>("maxiter", root.maxstep);
            reader.requireTagEnd();
        } else if (param == "stripe-root") {
            stripe_root.tolx = reader.getAttribute<double>("tolx", stripe_root.tolx);
            stripe_root.tolf_min = reader.getAttribute<double>("tolf-min", stripe_root.tolf_min);
            stripe_root.tolf_max = reader.getAttribute<double>("tolf-max", stripe_root.tolf_max);
            stripe_root.maxstep = reader.getAttribute<double>("maxstep", stripe_root.maxstep);
            stripe_root.maxiter = reader.getAttribute<int>("maxiter", stripe_root.maxiter);
            reader.requireTagEnd();
        } else if (param == "outer") {
            outdist = reader.requireAttribute<double>("distance");
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <mode>, <root>, <stripe_root>, or <outer>");
    }
}


dcomplex EffectiveFrequencyCylSolver::computeMode(dcomplex lambda)
{
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = %1%", str(lambda));
    if (isnan(k0.real())) k0 = 2e3*M_PI / lambda;
    stageOne();
    v = RootDigger(*this, [this](const dcomplex& v){return this->detS(v);}, log_value, root)(0.);
    dcomplex k = k0 * (1. - v/2.); // get modal frequency back from frequency parameter
    dcomplex lam = 2e3*M_PI / k;
    outWavelength = real(lam);
    outModalLoss = 1e7 * imag(k);
    outWavelength.fireChanged();
    outModalLoss.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return lam;
}


std::vector<dcomplex> EffectiveFrequencyCylSolver::findModes(dcomplex lambda1, dcomplex lambda2, size_t resteps, size_t imsteps, dcomplex eps)
{
    stageOne();
    outWavelength.invalidate();
    if (isnan(k0.real())) k0 = 4e3*M_PI / (lambda1 + lambda2);

    if ((real(lambda1) == 0. && real(lambda2) != 0.) || (real(lambda1) != 0. && real(lambda2) == 0.))
        throw BadInput(getId(), "Bad area to browse specified");

    dcomplex v0 =  2. - 4e3*M_PI / lambda1 / k0;
    dcomplex v1 =  2. - 4e3*M_PI / lambda1 / k0;

    if (eps.imag() == 0.) eps.imag(eps.real());

    if (real(eps) <= 0. || imag(eps) <= 0.)
        throw BadInput(this->getId(), "Bad precision specified");

    double re0 = real(v0), im0 = imag(v0);
    double re1 = real(v1), im1 = imag(v1);
    if (re0 > re1) std::swap(re0, re1);
    if (im0 > im1) std::swap(im0, im1);

    if (real(lambda1) == 0. && real(lambda2) == 0.) {
        re0 = 1e30;
        re1 = -1e30;
        for (size_t i = 0; i != rsize; ++i) {
            dcomplex v = veffs[i];
            if (v.real() < re0) re0 = v.real();
            if (v.real() > re1) re1 = v.real();
        }
    }
    if (imag(lambda1) && imag(lambda2)) {
        im0 = 1e30;
        im1 = -1e30;
        for (size_t i = 0; i != rsize; ++i) {
            dcomplex v = veffs[i];
            if (v.imag() < im0) im0 = v.imag();
            if (v.imag() > im1) im1 = v.imag();
        }
    }
    v0 = dcomplex(re0,im0);
    v1 = dcomplex(re1,im1);

    auto results = findZeros(this, [this](dcomplex z){return this->detS(z);}, v0, v1, resteps, imsteps, eps);

    if (results.size() != 0) {
        Data2DLog<dcomplex,dcomplex> logger(getId(), "freq", "v", "det");
        std::string msg = "Found modes at: ";
        for (auto& z: results) {
            dcomplex k = k0 * (1. - v/2.); // get modal frequency back from frequency parameter
            dcomplex lam = 2e3*M_PI / k;
            msg += str(lam) + ", ";
            logger(lam, detS(z));
            z = lam;
        }
        writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
    } else
        writelog(LOG_RESULT, "Did not find any modes");

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
    if (det > root.tolf_max) writelog(LOG_WARNING, "Provided wavelength does not correspond to any mode (det = %1%)", det);
    writelog(LOG_INFO, "Setting current mode to %1%", str(clambda));
    outWavelength = real(clambda);
    outModalLoss = 1e7 * imag(2e3*M_PI/clambda);
    outWavelength.fireChanged();
    outModalLoss.fireChanged();
    outIntensity.fireChanged();
}


void EffectiveFrequencyCylSolver::onInitialize()
{
    if (!geometry) throw NoGeometryException(getId());

    // Set default mesh
    if (!mesh) setSimpleMesh();

    // Assign space for refractive indices cache and stripe effective indices
    rsize = mesh->axis0.size();
    zsize = mesh->axis1.size() + 1;
    zbegin = 0;

    if (geometry->isExtended(Geometry::DIRECTION_VERT, false) &&
        abs(mesh->axis1[0] - geometry->getChild()->getBoundingBox().lower.c1) < SMALL)
        zbegin = 1;
    if (geometry->isExtended(Geometry::DIRECTION_TRAN, true) &&
        abs(mesh->axis0[mesh->axis0.size()-1] - geometry->getChild()->getBoundingBox().upper.c0) < SMALL)
        --rsize;
    if (geometry->isExtended(Geometry::DIRECTION_VERT, true) &&
        abs(mesh->axis1[mesh->axis1.size()-1] - geometry->getChild()->getBoundingBox().upper.c1) < SMALL)
        --zsize;

    nrCache.assign(rsize, std::vector<dcomplex,aligned_allocator<dcomplex>>(zsize));
    ngCache.assign(rsize, std::vector<dcomplex,aligned_allocator<dcomplex>>(zsize));
    veffs.resize(rsize);
    nng.resize(rsize);

    rfields.resize(rsize);
    zfields.resize(zsize);
}


void EffectiveFrequencyCylSolver::onInvalidate()
{
    outWavelength.invalidate();
    outModalLoss.invalidate();
    have_fields = false;
    outWavelength.fireChanged();
    outModalLoss.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/

static const int mh = 2; // Hankel function type (1 or 2)


void EffectiveFrequencyCylSolver::stageOne()
{
    bool fresh = !initCalculation();

    // Some additional checks
    for (auto x: mesh->axis0) {
        if (x < 0.) throw BadMesh(getId(), "for cylindrical geometry no radial points can be negative");
    }
    if (abs(mesh->axis0[0]) > SMALL) throw BadMesh(getId(), "radial mesh must start from zero");

    if (fresh || inTemperature.changed || inGain.changed || m != old_m || k0 != old_k0) { // we need to update something

        old_m = m;
        old_k0 = k0;

        double lam = real(2e3*M_PI / k0);

        writelog(LOG_DEBUG, "Updating refractive indices cache");

        double h = 1e6 * sqrt(SMALL);
        double lam1 = lam - h, lam2 = lam + h;
        double ih2 = 0.5 / h;

        RectilinearMesh2D midmesh = *mesh->getMidpointsMesh();
        if (rsize == mesh->axis0.size())
            midmesh.axis0.addPoint(mesh->axis0[mesh->axis0.size()-1] + outdist);
        if (zbegin == 0)
            midmesh.axis1.addPoint(mesh->axis1[0] - outdist);
        if (zsize == mesh->axis1.size()+1)
            midmesh.axis1.addPoint(mesh->axis1[mesh->axis1.size()-1] + outdist);

        auto temp = inTemperature(midmesh);
        auto gain = inGain(midmesh, lam1);
        auto gain_slope = inGain(midmesh, lam2).claim();
        {
            auto g1 = gain_slope.begin();
            auto g2 = gain.begin();
            for (; g1 != gain_slope.end(); ++g1, ++g2) *g1 = (*g2 - *g1) * ih2;
        }
        gain = inGain(midmesh, lam);

        for (size_t ir = 0; ir != rsize; ++ir) {
            for (size_t iz = zbegin; iz < zsize; ++iz) {
                size_t idx = midmesh.index(ir, iz-zbegin);
                double T = temp[idx];
                auto point = midmesh[idx];
                auto material = geometry->getMaterial(point);
                auto roles = geometry->getRolesAt(point);

                // Nr = nr + i/(4π) λ g
                // Ng = Nr - λ dN/dλ = Nr - λ dn/dλ - i/(4π) λ^2 dg/dλ
                if (roles.find("QW") == roles.end() && roles.find("QD") == roles.end() && roles.find("gain") == roles.end()) {
                    nrCache[ir][iz] = material->Nr(lam, T);
                    ngCache[ir][iz] = nrCache[ir][iz] - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) * ih2;
                } else { // we ignore the material absorption as it should be considered in the gain already
                    double g = gain[idx];
                    double gs = gain_slope[idx];
                    double nr = real(material->Nr(lam, T));
                    double ng = real(nrCache[ir][iz] - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) * ih2);
                    nrCache[ir][iz] = dcomplex(nr, 7.95774715459e-09 * lam * g);
                    ngCache[ir][iz] = dcomplex(ng, isnan(gs)? 0. : - 7.95774715459e-09 * lam*lam * gs);
                }
            }
            if (zbegin != 0) {
                nrCache[ir][0] = nrCache[ir][1];
                ngCache[ir][0] = ngCache[ir][1];
            }
        }

        // Compute effective frequencies for all stripes
        std::exception_ptr error; // needed to handle exceptions from OMP loop
        #pragma omp parallel for
        for (size_t i = 0; i < rsize; ++i) {
            if (error != std::exception_ptr()) continue; // just skip loops after error
            try {
                writelog(LOG_DETAIL, "Computing effective frequency for vertical stripe %1%", i);
#               ifndef NDEBUG
                    std::stringstream nrgs; for (auto nr = nrCache[i].end(), ng = ngCache[i].end(); nr != nrCache[i].begin();) {
                        --nr; --ng;
                        nrgs << ", (" << str(*nr) << ")/(" << str(*ng) << ")";
                    }
                    writelog(LOG_DEBUG, "Nr/nG[%1%] = [%2% ]", i, nrgs.str().substr(1));
#               endif

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
                    RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i],ngCache[i]);}, log_stripe, stripe_root);
                    veffs[i] = rootdigger(1e-3);
                    computeStripeNNg(i);
                }
            } catch (...) {
                #pragma omp critical
                error = std::current_exception();
            }
        }
        if (error != std::exception_ptr()) std::rethrow_exception(error);

#       ifndef NDEBUG
            std::stringstream strv; for (size_t i = 0; i < veffs.size(); ++i) strv << ", " << str(veffs[i]);
            writelog(LOG_DEBUG, "stripes veffs = [%1% ]", strv.str().substr(1));
            std::stringstream strn; for (size_t i = 0; i < nng.size(); ++i) strn << ", " << str(nng[i]);
            writelog(LOG_DEBUG, "stripes <nng> = [%1% ]", strn.str().substr(1));
#       endif
    }
}


dcomplex EffectiveFrequencyCylSolver::detS1(const dcomplex& v, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR,
                                            const std::vector<dcomplex,aligned_allocator<dcomplex>>& NG, bool save)
{
    double maxff = 0.;
    if (save) zfields[zbegin] = FieldZ(0., 1.);

    std::vector<dcomplex> kz(zsize);
    for (size_t i = zbegin; i < zsize; ++i) {
        kz[i] = k0 * sqrt(NR[i]*NR[i] - v * NR[i]*NG[i]);
        if (real(kz[i]) < 0.) kz[i] = -kz[i];
    }

    dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S

    dcomplex phas = 1.;
    if (zbegin != 0)
        phas = exp(I * kz[zbegin] * (mesh->axis1[zbegin]-mesh->axis1[zbegin-1]));

    for (size_t i = zbegin+1; i < zsize; ++i) {
        // Compute shift inside one layer
        s1 *= phas;
        s3 *= phas * phas;
        s4 *= phas;
        // Compute matrix after boundary
        dcomplex p = 0.5 + 0.5 * kz[i] / kz[i-1];
        dcomplex m = 1.0 - p;
        dcomplex chi = 1. / (p - m * s3);
        // F0 = [ (-m*m + p*p)*chi*s1  m*s1*s4*chi + s2 ] [ F2 ]
        // B2 = [ (-m + p*s3)*chi      s4*chi           ] [ B0 ]
        s2 += s1*m*chi*s4;
        s1 *= (p*p - m*m) * chi;
        s3  = (p*s3-m) * chi;
        s4 *= chi;
        // Compute phase shift for the next step
        if (i != mesh->axis1.size())
            phas = exp(I * kz[i] * (mesh->axis1[i]-mesh->axis1[i-1]));

        // Compute fields
        if (save) {
            dcomplex F = -s2/s1, B = (s1*s4-s2*s3)/s1;    // Assume  F0 = 0  B0 = 1
            double aFF = abs2(F), aBB = abs2(B);
            // zero very small fields to avoid errors in plotting for long layers
            if (aFF < 1e-16 * aBB) F = 0.; else maxff = max(maxff, aFF);
            if (aBB < 1e-16 * aFF) B = 0.; else maxff = max(maxff, aBB);
            zfields[i] = FieldZ(F, B);
        }
    }

    if (save) {
        zfields[zsize-1].B = 0.;
        maxff = 1. / sqrt(maxff);
        for (size_t i = zbegin; i < zsize; ++i) zfields[i] *= maxff;
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (size_t i = zbegin; i < zsize; ++i)
//                 nrs << "), (" << str(zfields[i].F) << ":" << str(zfields[i].B);
//             writelog(LOG_DEBUG, "vertical fields = [%1%) ]", nrs.str().substr(2));
//         }
// #endif
    }

    return s4 - s2*s3/s1;
}

std::vector<double,aligned_allocator<double>> EffectiveFrequencyCylSolver::computeWeights(size_t stripe)
{
    dcomplex veff = veffs[stripe];

    // Compute fields
    detS1(veff, nrCache[stripe], ngCache[stripe], true);

    std::vector<double,aligned_allocator<double>> weights(zsize);
    weights[zbegin] = 0.;
    weights[zsize-1] = 0.;
    double sum = weights[zbegin] + weights[zsize-1];

    for (size_t i = zbegin+1; i < zsize-1; ++i) {
        double d = mesh->axis1[i]-mesh->axis1[i-1];
        dcomplex kz = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - veff * nrCache[stripe][i]*ngCache[stripe][i]);
        if (real(kz) < 0.) kz = -kz;
        dcomplex w_ff, w_bb, w_fb, w_bf;
        if (d != 0.) {
            if (abs(imag(kz)) > SMALL) {
                dcomplex kk = kz - conj(kz);
                w_ff =   (exp(-I*d*kk) - 1.) / kk;
                w_bb = - (exp(+I*d*kk) - 1.) / kk;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            if (abs(real(kz)) > SMALL) {
                dcomplex kk = kz + conj(kz);
                w_fb =   (exp(-I*d*kk) - 1.) / kk;
                w_bf = - (exp(+I*d*kk) - 1.) / kk;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            dcomplex weight = zfields[i].F * conj(zfields[i].F) * w_ff +
                              zfields[i].F * conj(zfields[i].B) * w_fb +
                              zfields[i].B * conj(zfields[i].F) * w_bf +
                              zfields[i].B * conj(zfields[i].B) * w_bb;
            weights[i] = -imag(weight);
        } else
            weights[i] = 0.;
        sum += weights[i];
    }

    sum = 1. / sum;
    for (size_t i = zbegin; i < zsize; ++i) {
        weights[i] *= sum;
    }
// #ifndef NDEBUG
//     {
//         std::stringstream nrs; for (size_t i = zbegin; i < zsize; ++i) nrs << ", " << str(weights[i]);
//         writelog(LOG_DEBUG, "vertical weights = [%1%) ]", nrs.str().substr(2));
//     }
// #endif

    return std::move(weights);
}

void EffectiveFrequencyCylSolver::computeStripeNNg(size_t stripe)
{
    std::vector<double,aligned_allocator<double>> weight = computeWeights(stripe);
    nng[stripe] = 0.;
    for (size_t i = zbegin; i < zsize; ++i) {
        nng[stripe] += weight[i] * nrCache[stripe][i] * ngCache[stripe][i];
    }
}


dcomplex EffectiveFrequencyCylSolver::detS(const dcomplex& v)
{
    // In the outermost layer, there is only an outgoing wave, so the solution is only the Hankel function
    rfields[rsize-1] = FieldR(0., 1.);

    for (size_t i = rsize-1; i > 0; --i) {

        double r = mesh->axis0[i];

        dcomplex x1 = r * k0 * sqrt(nng[i-1] * (veffs[i-1]-v));
        if (real(x1) < 0.) x1 = -x1;

        dcomplex x2 = r * k0 * sqrt(nng[i] * (veffs[i]-v));
        if (real(x2) < 0.) x2 = -x2;

        // Compute Bessel functions and their derivatives
        dcomplex J1[2], H1[2];
        dcomplex J2[2], H2[2];
        double Jr[2], Ji[2], Hr[2], Hi[2];
        long nz, ierr;

        zbesj(x1.real(), x1.imag(), m, 1, 2, Jr, Ji, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", m, str(x1));
        zbesh(x1.real(), x1.imag(), m, 1, mh, 2, Hr, Hi, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", m, str(x1));
        for (int i = 0; i < 2; ++i) { J1[i] = dcomplex(Jr[i], Ji[i]); H1[i] = dcomplex(Hr[i], Hi[i]); }

        zbesj(x2.real(), x2.imag(), m, 1, 2, Jr, Ji, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", m, str(x2));
        zbesh(x2.real(), x2.imag(), m, 1, mh, 2, Hr, Hi, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", m, str(x2));
        for (int i = 0; i < 2; ++i) { J2[i] = dcomplex(Jr[i], Ji[i]); H2[i] = dcomplex(Hr[i], Hi[i]); }


        MatrixR A(  J1[0],                 H1[0],
                  m*J1[0] - x1*J1[1],    m*H1[0] - x1*H1[1]);

        MatrixR B(  J2[0],                 H2[0],
                  m*J2[0] - x2*J2[1],    m*H2[0] - x2*H2[1]);

        rfields[i-1] = A.solve(B * rfields[i]);
    }

    // In the innermost area there must not be any infinity, so H = 0.
    //return rfields[0].H / rfields[0].J; // to stress the difference between J and H
    return rfields[0].H;
}



plask::DataVector<const double> EffectiveFrequencyCylSolver::getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (!outWavelength.hasValue() || k0 != old_k0 || m != old_m) throw NoValue(OpticalIntensity::NAME);

    if (!have_fields) {
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = 0; i < rsize; ++i)
                nrs << "), (" << str(rfields[i].J) << ":" << str(rfields[i].H);
            writelog(LOG_DEBUG, "horizontal fields = [%1%) ]", nrs.str().substr(2));
        }
#endif

        stripe = 0;
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
        detS1(veffs[stripe], nrCache[stripe], ngCache[stripe], true);

        have_fields = true;
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
            long nz, ierr;

            size_t ir = mesh->axis0.findIndex(r); if (ir > 0) --ir; if (ir >= veffs.size()) ir = veffs.size()-1;
            dcomplex x = r * k0 * sqrt(nng[ir] * (veffs[ir]-v));
            if (real(x) < 0.) x = -x;
            zbesj(x.real(), x.imag(), m, 1, 1, &Jr, &Ji, nz, ierr);
            if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", m, str(x));
            if (ir == 0) {
                Hr = Hi = 0.;
            } else {
                zbesh(x.real(), x.imag(), m, 1, mh, 1, &Hr, &Hi, nz, ierr);
                if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", m, str(x));
            }
            dcomplex val = rfields[ir].J * dcomplex(Jr, Ji) + rfields[ir].H * dcomplex(Hr, Hi);

            size_t iz = mesh->axis1.findIndex(z);
            if (iz >= zsize) iz = zsize-1;
            dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - v * nrCache[stripe][iz]*ngCache[stripe][iz]);
            if (real(kz) < 0.) kz = -kz;
            z -= mesh->axis1[max(int(iz)-1, 0)];
            dcomplex phasz = exp(- I * kz * z);
            val *= zfields[iz].F * phasz + zfields[iz].B / phasz;

            results[id] = abs2(val);
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

        bool error = false; // needed to handle exceptions from OMP loop
        std::string errormsg;
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (size_t idr = 0; idr < rect_mesh.tran().size(); ++idr) {
                if (error) continue;
                double r = rect_mesh.axis0[idr];
                double Jr, Ji, Hr, Hi;
                long nz, ierr;
                if (r < 0.) r = -r;
                size_t ir = mesh->axis0.findIndex(r); if (ir > 0) --ir; if (ir >= veffs.size()) ir = veffs.size()-1;
                dcomplex x = r * k0 * sqrt(nng[ir] * (veffs[ir]-v));
                if (real(x) < 0.) x = -x;
                zbesj(x.real(), x.imag(), m, 1, 1, &Jr, &Ji, nz, ierr);
                if (ierr != 0)
                #pragma omp critical
                {
                    error = true;
                    errormsg = format("Could not compute J(%1%, %2%)", m, str(x));
                }
                if (ir == 0) {
                    Hr = Hi = 0.;
                } else {
                    zbesh(x.real(), x.imag(), m, 1, mh, 1, &Hr, &Hi, nz, ierr);
                    if (ierr != 0)
                    #pragma omp critical
                    {
                        error = true;
                        errormsg = format("Could not compute H(%1%, %2%)", m, str(x));
                    }
                }
                valr[idr] = rfields[ir].J * dcomplex(Jr, Ji) + rfields[ir].H * dcomplex(Hr, Hi);
            }

            if (!error) {
                #pragma omp for
                for (size_t idz = 0; idz < rect_mesh.vert().size(); ++idz) {
                    double z = rect_mesh.axis1[idz];
                    size_t iz = mesh->axis1.findIndex(z);
                    if (iz >= zsize) iz = zsize-1;
                    dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - v * nrCache[stripe][iz]*ngCache[stripe][iz]);
                    if (real(kz) < 0.) kz = -kz;
                    z -= mesh->axis1[max(int(iz)-1, 0)];
                    dcomplex phasz = exp(- I * kz * z);
                    valz[idz] = zfields[iz].F * phasz + zfields[iz].B / phasz;
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
        }

        if (error)
            throw ComputationError(getId(), errormsg);

        return true;
    }

    return false;
}


}}} // namespace plask::solvers::effective
