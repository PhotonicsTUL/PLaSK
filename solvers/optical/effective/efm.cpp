#include "efm.h"
#include "patterson.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

EffectiveFrequencyCylSolver::EffectiveFrequencyCylSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCylindrical, RectilinearMesh2D>(name),
    log_value(dataLog<dcomplex, dcomplex>("freq", "v", "det")),
    emission(TOP),
    outdist(0.1),
    perr(1e-3),
    k0(NAN),
    outWavelength(this, &EffectiveFrequencyCylSolver::getWavelength, &EffectiveFrequencyCylSolver::nmodes),
    outLoss(this, &EffectiveFrequencyCylSolver::getModalLoss,  &EffectiveFrequencyCylSolver::nmodes),
    outLightIntensity(this, &EffectiveFrequencyCylSolver::getLightIntenisty,  &EffectiveFrequencyCylSolver::nmodes) {
    inTemperature = 300.;
    root.tolx = 1.0e-8;
    root.tolf_min = 1.0e-10;
    root.tolf_max = 2.0e-5;
    root.maxstep = 0.1;
    root.maxiter = 500;
    stripe_root.tolx = 1.0e-8;
    stripe_root.tolf_min = 1.0e-10;
    stripe_root.tolf_max = 1.0e-5;
    stripe_root.maxstep = 0.05;
    stripe_root.maxiter = 500;
}


void EffectiveFrequencyCylSolver::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
            // m = reader.getAttribute<unsigned short>("m", m);
            auto alam0 = reader.getAttribute<double>("lam0");
            auto ak0 = reader.getAttribute<double>("k0");
            if (alam0) {
                if (ak0) throw XMLConflictingAttributesException(reader, "k0", "lam0");
                k0 = 2e3*M_PI / *alam0;
            } else if (ak0) k0 = *ak0;
            emission = reader.enumAttribute<Emission>("emission").value("top", TOP).value("bottom", BOTTOM).get(emission);
            reader.requireTagEnd();
        } else if (param == "root") {
            root.tolx = reader.getAttribute<double>("tolx", root.tolx);
            root.tolf_min = reader.getAttribute<double>("tolf-min", root.tolf_min);
            root.tolf_max = reader.getAttribute<double>("tolf-max", root.tolf_max);
            root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
            root.maxiter = reader.getAttribute<int>("maxiter", root.maxiter);
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
        } else if (param == "mesh") {
            auto name = reader.getAttribute("ref");
            if (!name) name.reset(reader.requireTextInCurrentTag());
            else reader.requireTagEnd();
            auto found = manager.meshes.find(*name);
            if (found != manager.meshes.end()) {
                auto mesh1 = dynamic_pointer_cast<RectilinearMesh1D>(found->second);
                auto mesh2 = dynamic_pointer_cast<RectilinearMesh2D>(found->second);
                if (mesh1) this->setHorizontalMesh(mesh1->axis);
                else if (mesh2) this->setMesh(mesh2);
                else throw BadInput(this->getId(), "Mesh '%1%' of wrong type", *name);
            } else {
                auto found = manager.generators.find(*name);
                if (found != manager.generators.end()) {
                    auto generator1 = dynamic_pointer_cast<MeshGeneratorOf<RectilinearMesh1D>>(found->second);
                    auto generator2 = dynamic_pointer_cast<MeshGeneratorOf<RectilinearMesh2D>>(found->second);
                    if (generator1) this->setMesh(make_shared<RectilinearMesh2DFrom1DGenerator>(generator1));
                    else if (generator2) this->setMesh(generator2);
                    else throw BadInput(this->getId(), "Mesh generator '%1%' of wrong type", *name);
                } else
                    throw BadInput(this->getId(), "Neither mesh nor mesh generator '%1%' found", *name);
            }
        } else
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <mode>, <root>, <stripe_root>, or <outer>");
    }
}


size_t EffectiveFrequencyCylSolver::findMode(dcomplex lambda, int m)
{
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = %1%", str(lambda));
    if (isnan(k0.real())) k0 = 2e3*M_PI / lambda;
    dcomplex v = 2. - 4e3*M_PI / lambda / k0;
    stageOne();
    Mode mode(this, m);
    mode.freqv = RootDigger(*this, [this,&mode](const dcomplex& v){return this->detS(v,mode);}, log_value, root)(v);
    return insertMode(mode);
}


std::vector<size_t> EffectiveFrequencyCylSolver::findModes(dcomplex lambda1, dcomplex lambda2, int m, size_t resteps, size_t imsteps, dcomplex eps)
{
    if (isnan(k0.real())) k0 = 4e3*M_PI / (lambda1 + lambda2);
    stageOne();

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
    if (imag(lambda1) == 0. && imag(lambda2) == 0.) {
        im0 = 1e30;
        im1 = -1e30;
        for (size_t i = 0; i != rsize; ++i) {
            dcomplex v = veffs[i];
            if (v.imag() < im0) im0 = v.imag();
            if (v.imag() > im1) im1 = v.imag();
        }
    }
    v0 = 1.000001 * dcomplex(re0,im0);
    v1 = 0.999999 * dcomplex(re1,im1);

    Mode mode(this, m);
    auto results = findZeros(this, [this,&mode](dcomplex v){return this->detS(v,mode);}, v0, v1, resteps, imsteps, eps);

    std::vector<size_t> idx(results.size());

    if (results.size() != 0) {
        Data2DLog<dcomplex,dcomplex> logger(getId(), "freq", "v", "det");
        RootDigger refine(*this, [this,&mode](const dcomplex& v){return this->detS(v,mode);}, logger, root);
        std::string msg = "Found modes at: ";
        for (auto& z: results) {
            try {
                z = refine(z);
            } catch (ComputationError) {
                continue;
            }
            mode.freqv = z;
            idx.push_back(insertMode(mode));
            dcomplex k = k0 * (1. - z/2.); // get modal frequency back from frequency parameter
            dcomplex lam = 2e3*M_PI / k;
            msg += str(lam) + ", ";
        }
        writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
    } else
        writelog(LOG_RESULT, "Did not find any modes");

    return idx;
}


size_t EffectiveFrequencyCylSolver::setMode(dcomplex clambda, int m)
{
    if (isnan(k0.real())) k0 = 2e3*M_PI / clambda;
    if (!initialized) {
        writelog(LOG_WARNING, "Solver invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    Mode mode(this, m);
    mode.freqv = 2. - 4e3*M_PI / clambda / k0;
    double det = abs(detS(mode.freqv, mode));
    if (det > root.tolf_max)
        writelog(LOG_WARNING, "Provided wavelength does not correspond to any mode (det = %1%)", det);
    writelog(LOG_INFO, "Setting mode at %1%", str(clambda));
    return insertMode(mode);
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

    zfields.resize(zsize);

    need_gain = false;
}


void EffectiveFrequencyCylSolver::onInvalidate()
{
    if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
    modes.clear();
    outWavelength.fireChanged();
    outLoss.fireChanged();
    outLightIntensity.fireChanged();
}

/********* Here are the computations *********/

void EffectiveFrequencyCylSolver::stageOne()
{
    bool fresh = !initCalculation();

    // Some additional checks
    for (auto x: mesh->axis0) {
        if (x < 0.) throw BadMesh(getId(), "for cylindrical geometry no radial points can be negative");
    }
    if (abs(mesh->axis0[0]) > SMALL) throw BadMesh(getId(), "radial mesh must start from zero");

    if (fresh || inTemperature.changed() || (need_gain && inGain.changed()) || k0 != old_k0) { // we need to update something

        if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        modes.clear();

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
        bool have_gain = false;
        DataVector<const double> gain;
        DataVector<double> gain_slope;

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
                    need_gain = true;
                    if (!have_gain) {
                        gain = inGain(midmesh, lam1);
                        gain_slope = inGain(midmesh, lam2).claim();
                        auto g1 = gain_slope.begin();
                        auto g2 = gain.begin();
                        for (; g1 != gain_slope.end(); ++g1, ++g2) *g1 = (*g2 - *g1) * ih2;
                        gain = inGain(midmesh, lam);
                        have_gain = true;
                    }
                    double g = gain[idx];
                    double gs = gain_slope[idx];
                    double nr = real(material->Nr(lam, T));
                    double ng = real(nr - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) * ih2);
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
                    writelog(LOG_DEBUG, "Nr/Ng[%1%] = [%2% ]", i, nrgs.str().substr(1));
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
                    veffs[i] = rootdigger(1e-5);
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

        double rmin=INFINITY, rmax=-INFINITY, imin=INFINITY, imax=-INFINITY;
        for (auto v: veffs) {
            dcomplex lam = 2e3*M_PI / (k0 * (1. - v/2.));
            if (real(lam) < rmin) rmin = real(lam);
            if (real(lam) > rmax) rmax = real(lam);
            if (imag(lam) < imin) imin = imag(lam);
            if (imag(lam) > imax) imax = imag(lam);
        }
        writelog(LOG_DETAIL, "Wavelengths should be between %1%nm and %2%nm", str(dcomplex(rmin,imin)), str(dcomplex(rmax,imax)));
    }
}


dcomplex EffectiveFrequencyCylSolver::detS1(const dcomplex& v, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR,
                                            const std::vector<dcomplex,aligned_allocator<dcomplex>>& NG, std::vector<FieldZ>* saveto)
{
    if (saveto) (*saveto)[zbegin] = FieldZ(0., 1.);

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
        if (saveto) {
            dcomplex F = -s2/s1, B = (s1*s4-s2*s3)/s1;    // Assume  F0 = 0  B0 = 1
            double aF = abs(F), aB = abs(B);
            // zero very small fields to avoid errors in plotting for long layers
            if (aF < 1e-8 * aB) F = 0.;
            if (aB < 1e-8 * aF) B = 0.;
            (*saveto)[i] = FieldZ(F, B);
        }
    }

    if (saveto) {
        if (emission == TOP) {
            dcomplex f = 1. / (*saveto)[zsize-1].F;
            (*saveto)[zsize-1] = FieldZ(1., 0.);
            for (size_t i = zbegin; i < zsize-1; ++i) (*saveto)[i] *= f;
        } else {
            (*saveto)[zsize-1].B = 0.;
            // we dont have to scale fields as we have already set (*saveto)[zbegin] = FieldZ(0., 1.)
        }
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (size_t i = zbegin; i < zsize; ++i)
//                 nrs << "), (" << str((*saveto)[i].F) << ":" << str((*saveto)[i].B);
//             writelog(LOG_DEBUG, "vertical fields = [%1%) ]", nrs.str().substr(2));
//         }
// #endif
    }

    return s4 - s2*s3/s1;
}

void EffectiveFrequencyCylSolver::computeStripeNNg(size_t stripe)
{
    nng[stripe] = 0.;

    std::vector<FieldZ> zfield(zsize);

    dcomplex veff = veffs[stripe];

    // Compute fields
    detS1(veff, nrCache[stripe], ngCache[stripe], &zfield);

    double sum = 0.;

    for (size_t i = zbegin+1; i < zsize-1; ++i) {
        double d = mesh->axis1[i]-mesh->axis1[i-1];
        double weight = 0.;
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
            weight -= imag(zfield[i].F * conj(zfield[i].F) * w_ff +
                           zfield[i].F * conj(zfield[i].B) * w_fb +
                           zfield[i].B * conj(zfield[i].F) * w_bf +
                           zfield[i].B * conj(zfield[i].B) * w_bb);
        }
        sum += weight;
        nng[stripe] += weight * nrCache[stripe][i] * ngCache[stripe][i];
    }

    nng[stripe] /= sum;
}

double EffectiveFrequencyCylSolver::integrateBessel(const Mode& mode) const
{
    double err = perr;
    double intr = patterson<double>([this,&mode](double r){return r * abs2(mode.rField(r));}, 0., 2.*mesh->axis0[rsize-1], err);
    return 2*M_PI * intr; //TODO consider m <> 0
}

dcomplex EffectiveFrequencyCylSolver::detS(const dcomplex& v, plask::solvers::effective::EffectiveFrequencyCylSolver::Mode& mode, bool save)
{
    // In the outermost layer, there is only an outgoing wave, so the solution is only the Hankel function
    mode.rfields[rsize-1] = FieldR(0., 1.);

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

        zbesj(x1.real(), x1.imag(), mode.m, 1, 2, Jr, Ji, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", mode.m, str(x1));
        zbesh(x1.real(), x1.imag(), mode.m, 1, MH, 2, Hr, Hi, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", mode.m, str(x1));
        for (int j = 0; j < 2; ++j) { J1[j] = dcomplex(Jr[j], Ji[j]); H1[j] = dcomplex(Hr[j], Hi[j]); }

        zbesj(x2.real(), x2.imag(), mode.m, 1, 2, Jr, Ji, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)", mode.m, str(x2));
        zbesh(x2.real(), x2.imag(), mode.m, 1, MH, 2, Hr, Hi, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)", mode.m, str(x2));
        for (int j = 0; j < 2; ++j) { J2[j] = dcomplex(Jr[j], Ji[j]); H2[j] = dcomplex(Hr[j], Hi[j]); }

        MatrixR A(       J1[0],                   H1[0],
                  mode.m*J1[0] - x1*J1[1], mode.m*H1[0] - x1*H1[1]);

        MatrixR B(       J2[0],                   H2[0],
                  mode.m*J2[0] - x2*J2[1], mode.m*H2[0] - x2*H2[1]);

        mode.rfields[i-1] = A.solve(B * mode.rfields[i]);
    }

    if (save) {
        register dcomplex f = sqrt(1e12 / integrateBessel(mode));
        for (size_t r = 0; r != rsize; ++r) mode.rfields[r] *= f;
    }

    // In the innermost area there must not be any infinity, so H = 0.
    //return rfields[0].H / rfields[0].J; // to stress the difference between J and H
    return mode.rfields[0].H;
}



plask::DataVector<const double> EffectiveFrequencyCylSolver::getLightIntenisty(int num, const MeshD<2>& dst_mesh, InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (modes.size() <= num || k0 != old_k0) throw NoValue(LightIntensity::NAME);

    if (!modes[num].have_fields) {
        detS(modes[num].freqv, modes[num], true);
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = 0; i < rsize; ++i)
                nrs << "), (" << str(modes[num].rfields[i].J) << ":" << str(modes[num].rfields[i].H);
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
            if (all_the_same) ++stripe;
        }
        writelog(LOG_DETAIL, "Vertical field distribution taken from stripe %1%", stripe);

        // Compute vertical part
        detS1(veffs[stripe], nrCache[stripe], ngCache[stripe], &zfields);

        modes[num].have_fields = true;
    }

    DataVector<double> results(dst_mesh.size());

    if (!getLightIntenisty_Efficient<RectilinearMesh2D>(num, dst_mesh, results) &&
        !getLightIntenisty_Efficient<RegularMesh2D>(num, dst_mesh, results)) {

        std::exception_ptr error; // needed to handle exceptions from OMP loop

        double power = 1e-3 * modes[num].power; // 1e-3 mW->W

        #pragma omp parallel for schedule(static,1024)
        for (size_t id = 0; id < dst_mesh.size(); ++id) {
            if (error) continue;

            auto point = dst_mesh[id];
            double r = point.c0;
            double z = point.c1;
            if (r < 0) r = -r;

            dcomplex val;
            try {
                val = modes[num].rField(r);
            } catch (...) {
                #pragma omp critical
                error = std::current_exception();
            }

            size_t iz = mesh->axis1.findIndex(z);
            if (iz >= zsize) iz = zsize-1;
            else if (iz < zbegin) iz = zbegin;
            dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - veffs[stripe] * nrCache[stripe][iz]*ngCache[stripe][iz]);
            if (real(kz) < 0.) kz = -kz;
            z -= mesh->axis1[max(int(iz)-1, 0)];
            dcomplex phasz = exp(- I * kz * z);
            val *= zfields[iz].F * phasz + zfields[iz].B / phasz;

            results[id] = power * abs2(val);
        }

        if (error) std::rethrow_exception(error);
    }

    return results;
}

template <typename MeshT>
bool EffectiveFrequencyCylSolver::getLightIntenisty_Efficient(size_t num, const MeshD<2>& dst_mesh, DataVector<double>& results)
{
    if (dynamic_cast<const MeshT*>(&dst_mesh)) {

        const MeshT& rect_mesh = dynamic_cast<const MeshT&>(dst_mesh);

        std::vector<dcomplex> valr(rect_mesh.axis0.size());
        std::vector<dcomplex> valz(rect_mesh.axis1.size());

        std::exception_ptr error; // needed to handle exceptions from OMP loop

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (size_t idr = 0; idr < rect_mesh.tran().size(); ++idr) {
                if (error) continue;
                double r = rect_mesh.axis0[idr];
                if (r < 0.) r = -r;
                try {
                    valr[idr] = modes[num].rField(r);
                } catch (...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }

            if (!error) {
                #pragma omp for
                for (size_t idz = 0; idz < rect_mesh.vert().size(); ++idz) {
                    double z = rect_mesh.axis1[idz];
                    size_t iz = mesh->axis1.findIndex(z);
                    if (iz >= zsize) iz = zsize-1;
                    else if (iz < zbegin) iz = zbegin;
                    dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - veffs[stripe] * nrCache[stripe][iz]*ngCache[stripe][iz]);
                    if (real(kz) < 0.) kz = -kz;
                    z -= mesh->axis1[max(int(iz)-1, 0)];
                    dcomplex phasz = exp(- I * kz * z);
                    valz[idz] = zfields[iz].F * phasz + zfields[iz].B / phasz;
                }

                double power = 1e-3 * modes[num].power; // 1e-3 mW->W

                if (rect_mesh.getIterationOrder() == MeshT::ORDER_NORMAL) {
                    #pragma omp for
                    for (size_t i1 = 0; i1 < rect_mesh.axis1.size(); ++i1) {
                        double* data = results.data() + i1 * rect_mesh.axis0.size();
                        for (size_t i0 = 0; i0 < rect_mesh.axis0.size(); ++i0) {
                            dcomplex f = valr[i0] * valz[i1];
                            data[i0] = power * abs2(f);
                        }
                    }
                } else {
                    #pragma omp for
                    for (size_t i0 = 0; i0 < rect_mesh.axis0.size(); ++i0) {
                        double* data = results.data() + i0 * rect_mesh.axis1.size();
                        for (size_t i1 = 0; i1 < rect_mesh.axis1.size(); ++i1) {
                            dcomplex f = valr[i0] * valz[i1];
                            data[i1] = power * abs2(f);
                        }
                    }
                }
            }
        }

        if (error) std::rethrow_exception(error);

        return true;
    }

    return false;
}


}}} // namespace plask::solvers::effective
