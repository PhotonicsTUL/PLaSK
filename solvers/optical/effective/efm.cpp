#include "efm.h"
#include "patterson.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

#define DLAM 1e-3

EffectiveFrequencyCylSolver::EffectiveFrequencyCylSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCylindrical, RectangularMesh<2>>(name),
    log_value(dataLog<dcomplex, dcomplex>("radial", "lam", "det")),
    emission(TOP),
    rstripe(-1),
    outdist(0.1),
    perr(1e-3),
    k0(NAN),
    vlam(0.),
    outWavelength(this, &EffectiveFrequencyCylSolver::getWavelength, &EffectiveFrequencyCylSolver::nmodes),
    outLoss(this, &EffectiveFrequencyCylSolver::getModalLoss,  &EffectiveFrequencyCylSolver::nmodes),
    outLightMagnitude(this, &EffectiveFrequencyCylSolver::getLightMagnitude,  &EffectiveFrequencyCylSolver::nmodes),
    outRefractiveIndex(this, &EffectiveFrequencyCylSolver::getRefractiveIndex),
    outHeat(this, &EffectiveFrequencyCylSolver::getHeat) {
    inTemperature = 300.;
    root.tolx = 1.0e-6;
    root.tolf_min = 1.0e-7;
    root.tolf_max = 2.0e-5;
    root.maxiter = 500;
    stripe_root.tolx = 1.0e-6;
    stripe_root.tolf_min = 1.0e-7;
    stripe_root.tolf_max = 1.0e-5;
    stripe_root.maxiter = 500;
    inTemperature.changedConnectMethod(this, &EffectiveFrequencyCylSolver::onInputChange);
    inGain.changedConnectMethod(this, &EffectiveFrequencyCylSolver::onInputChange);
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
            vlam = reader.getAttribute<double>("vlam", real(vlam));
            if (reader.hasAttribute("vat")) {
                std::string str = reader.requireAttribute("vat");
                if (str == "all" || str == "none") rstripe = -1;
                else
                    try { setStripeR(boost::lexical_cast<double>(str)); }
                    catch (boost::bad_lexical_cast) { throw XMLBadAttrException(reader, "vat", str); }
            }
            reader.requireTagEnd();
        } else if (param == "root") {
            root.tolx = reader.getAttribute<double>("tolx", root.tolx);
            root.tolf_min = reader.getAttribute<double>("tolf-min", root.tolf_min);
            root.tolf_max = reader.getAttribute<double>("tolf-max", root.tolf_max);
            // root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
            root.maxiter = reader.getAttribute<int>("maxiter", root.maxiter);
            reader.requireTagEnd();
        } else if (param == "stripe-root") {
            stripe_root.tolx = reader.getAttribute<double>("tolx", stripe_root.tolx);
            stripe_root.tolf_min = reader.getAttribute<double>("tolf-min", stripe_root.tolf_min);
            stripe_root.tolf_max = reader.getAttribute<double>("tolf-max", stripe_root.tolf_max);
            // stripe_root.maxstep = reader.getAttribute<double>("maxstep", stripe_root.maxstep);
            stripe_root.maxiter = reader.getAttribute<int>("maxiter", stripe_root.maxiter);
            reader.requireTagEnd();
        } else if (param == "outer") {
            outdist = reader.requireAttribute<double>("dist");
            reader.requireTagEnd();
        } else if (param == "mesh") {
            auto name = reader.getAttribute("ref");
            if (!name) name.reset(reader.requireTextInCurrentTag());
            else reader.requireTagEnd();
            auto found = manager.meshes.find(*name);
            if (found != manager.meshes.end()) {
                auto mesh1 = dynamic_pointer_cast<RectangularAxis>(found->second);
                auto mesh2 = dynamic_pointer_cast<RectangularMesh<2>>(found->second);
                if (mesh1) this->setHorizontalMesh(mesh1);
                else if (mesh2) this->setMesh(mesh2);
                else throw BadInput(this->getId(), "Mesh '%1%' of wrong type", *name);
            } else {
                auto found = manager.generators.find(*name);
                if (found != manager.generators.end()) {
                    auto generator1 = dynamic_pointer_cast<MeshGeneratorD<1>>(found->second);
                    auto generator2 = dynamic_pointer_cast<MeshGeneratorD<2>>(found->second);
                    if (generator1) this->setMesh(make_shared<RectilinearMesh2DFrom1DGenerator>(generator1));
                    else if (generator2) this->setMesh(generator2);
                    else throw BadInput(this->getId(), "Mesh generator '%1%' of wrong type", *name);
                } else
                    throw BadInput(this->getId(), "Neither mesh nor mesh generator '%1%' found", *name);
            }
        } else
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <mode>, <root>, <stripe-root>, or <outer>");
    }
}


size_t EffectiveFrequencyCylSolver::findMode(dcomplex lambda, int m)
{
    writelog(LOG_INFO, "Searching for the mode starting from wavelength = %1%", str(lambda));
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
    stageOne();
    Mode mode(this, m);
    mode.lam = RootMuller(*this, [this,&mode](const dcomplex& lam){return this->detS(lam,mode);}, log_value, root)(lambda-DLAM, lambda+DLAM);
    return insertMode(mode);
}


size_t EffectiveFrequencyCylSolver::findMode(dcomplex lambda1, dcomplex lambda2, int m)
{
    writelog(LOG_INFO, "Searching for the mode between wavelengths %1% and %2%", str(lambda1), str(lambda2));
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
    stageOne();
    Mode mode(this, m);
    mode.lam = RootMuller(*this, [this,&mode](const dcomplex& lam){return this->detS(lam,mode);}, log_value, root)(lambda1, lambda2);
    return insertMode(mode);
}


std::vector<size_t> EffectiveFrequencyCylSolver::findModes(dcomplex lambda1, dcomplex lambda2, int m, size_t resteps, size_t imsteps, dcomplex eps)
{
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
    stageOne();

    if ((real(lambda1) == 0. && real(lambda2) != 0.) || (real(lambda1) != 0. && real(lambda2) == 0.))
        throw BadInput(getId(), "Bad area to browse specified");

    dcomplex lam0 =  lambda1;
    dcomplex lam1 =  lambda2;

    if (eps.imag() == 0.) eps.imag(eps.real());

    if (real(eps) <= 0. || imag(eps) <= 0.)
        throw BadInput(this->getId(), "Bad precision specified");

    double re0 = real(lam0), im0 = imag(lam0);
    double re1 = real(lam1), im1 = imag(lam1);
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
    lam0 = 1.000001 * dcomplex(re0,im0);
    lam1 = 0.999999 * dcomplex(re1,im1);

    Mode mode(this, m);
    auto results = findZeros(this, [this,&mode](dcomplex lam){return this->detS(lam,mode);}, lam0, lam1, resteps, imsteps, eps);

    std::vector<size_t> idx(results.size());

    if (results.size() != 0) {
        log_value.resetCounter();
        RootMuller refine(*this, [this,&mode](const dcomplex& lam){return this->detS(lam,mode);}, log_value, root);
        std::string msg = "Found modes at: ";
        for (auto& zz: results) {
            dcomplex z;
            try {
                z = refine(zz.first, zz.second);
            } catch (ComputationError) {
                continue;
            }
            mode.lam = z;
            idx.push_back(insertMode(mode));
            msg += str(z) + ", ";
        }
        writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
    } else
        writelog(LOG_RESULT, "Did not find any modes");

    return idx;
}


size_t EffectiveFrequencyCylSolver::setMode(dcomplex clambda, int m)
{
    if (isnan(k0.real())) throw BadInput(getId(), "No reference wavelength `lam0` specified");
    if (!initialized) {
        writelog(LOG_WARNING, "Solver invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    Mode mode(this, m);
    mode.lam = clambda;
    double det = abs(detS(mode.lam, mode));
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
    rsize = mesh->axis0->size();
    zsize = mesh->axis1->size() + 1;
    zbegin = 0;

    if (geometry->isExtended(Geometry::DIRECTION_VERT, false) &&
        abs(mesh->axis1->at(0) - geometry->getChild()->getBoundingBox().lower.c1) < SMALL)
        zbegin = 1;
    if (geometry->isExtended(Geometry::DIRECTION_TRAN, true) &&
        abs(mesh->axis0->at(mesh->axis0->size()-1) - geometry->getChild()->getBoundingBox().upper.c0) < SMALL)
        --rsize;
    if (geometry->isExtended(Geometry::DIRECTION_VERT, true) &&
        abs(mesh->axis1->at(mesh->axis1->size()-1) - geometry->getChild()->getBoundingBox().upper.c1) < SMALL)
        --zsize;

    nrCache.assign(rsize, std::vector<dcomplex,aligned_allocator<dcomplex>>(zsize));
    ngCache.assign(rsize, std::vector<dcomplex,aligned_allocator<dcomplex>>(zsize));
    veffs.resize(rsize);
    nng.resize(rsize);

    zfields.resize(zsize);

    need_gain = false;
    cache_outdated = true;
    have_veffs = false;
}


void EffectiveFrequencyCylSolver::onInvalidate()
{
    if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
    modes.clear();
    outWavelength.fireChanged();
    outLoss.fireChanged();
    outLightMagnitude.fireChanged();
}

/********* Here are the computations *********/

void EffectiveFrequencyCylSolver::updateCache()
{
    bool fresh = !initCalculation();

    if (fresh || cache_outdated || inTemperature.changed() || (need_gain && inGain.changed()) || k0 != old_k0) {
        // we need to update something

        // Some additional checks
        for (auto x: *mesh->axis0) {
            if (x < 0.) throw BadMesh(getId(), "for cylindrical geometry no radial points can be negative");
        }
        if (abs(mesh->axis0->at(0)) > SMALL) throw BadMesh(getId(), "radial mesh must start from zero");

        if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        modes.clear();

        old_k0 = k0;

        double lam = real(2e3*M_PI / k0);

        writelog(LOG_DEBUG, "Updating refractive indices cache");

        double h = 1e6 * sqrt(SMALL);
        double lam1 = lam - h, lam2 = lam + h;
        double i2h = 0.5 / h;

        shared_ptr<RectilinearAxis> axis0, axis1;
        {
            shared_ptr<RectangularMesh<2>> midmesh = mesh->getMidpointsMesh();
            axis0 = make_shared<RectilinearAxis>(*midmesh->axis0);
            axis1 = make_shared<RectilinearAxis>(*midmesh->axis1);
        }
        if (rsize == axis0->size())
            axis0->addPoint(axis0->at(axis0->size()-1) + outdist);
        if (zbegin == 0)
            axis1->addPoint(axis1->at(0) - outdist);
        if (zsize == mesh->axis1->size()+1)
            axis1->addPoint(mesh->axis1->at(mesh->axis1->size()-1) + outdist);

        RectangularMesh<2> midmesh(axis0, axis1, mesh->getIterationOrder());
        auto temp = inTemperature(midmesh);
        bool have_gain = false;
        DataVector<double> gain;
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
                    ngCache[ir][iz] = nrCache[ir][iz] - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) * i2h;
                } else { // we ignore the material absorption as it should be considered in the gain already
                    need_gain = true;
                    if (!have_gain) {
                        gain = inGain(midmesh, lam1).claim();
                        gain_slope = inGain(midmesh, lam2).claim();
                        auto g1 = gain.begin();
                        auto gs2 = gain_slope.begin();
                        for (; gs2 != gain_slope.end(); ++gs2, ++g1) {
                            double g = 0.5 * (*g1 + *gs2);
                            *gs2 = (*gs2 - *g1) * i2h;
                            *g1 = g;
                        }
                        have_gain = true;
                    }
                    double g = gain[idx];
                    double gs = gain_slope[idx];
                    double nr = real(material->Nr(lam, T));
                    double ng = real(nr - lam * (material->Nr(lam2, T) - material->Nr(lam1, T)) * i2h);
                    nrCache[ir][iz] = dcomplex(nr, (0.25e-7/M_PI) * lam * g);
                    ngCache[ir][iz] = dcomplex(ng, isnan(gs)? 0. : - (0.25e-7/M_PI) * lam*lam * gs);
                }
            }
            if (zbegin != 0) {
                nrCache[ir][0] = nrCache[ir][1];
                ngCache[ir][0] = ngCache[ir][1];
            }
        }
        cache_outdated = false;
        have_veffs = false;
    }
}

void EffectiveFrequencyCylSolver::stageOne()
{
    updateCache();

    if (!have_veffs) {
        if (rstripe < 0) {
            size_t main_stripe = getMainStripe();
            // Compute effective frequencies for all stripes
            std::exception_ptr error; // needed to handle exceptions from OMP loop
            #pragma omp parallel for
            for (size_t i = 0; i < rsize; ++i) {
                if (error != std::exception_ptr()) continue; // just skip loops after error
                try {
                    writelog(LOG_DETAIL, "Computing effective frequency for vertical stripe %1%", i);
#                   ifndef NDEBUG
                        std::stringstream nrgs; for (auto nr = nrCache[i].end(), ng = ngCache[i].end(); nr != nrCache[i].begin();) {
                            --nr; --ng;
                            nrgs << ", (" << str(*nr) << ")/(" << str(*ng) << ")";
                        }
                        writelog(LOG_DEBUG, "Nr/Ng[%1%] = [%2% ]", i, nrgs.str().substr(1));
#                   endif
                    dcomplex same_nr = nrCache[i].front();
                    dcomplex same_ng = ngCache[i].front();
                    bool all_the_same = true;
                    for (auto nr = nrCache[i].begin(), ng = ngCache[i].begin(); nr != nrCache[i].end(); ++nr, ++ng)
                        if (*nr != same_nr || *ng != same_ng) { all_the_same = false; break; }
                    if (all_the_same) {
                        veffs[i] = 1.; // TODO make sure this is so!
                        nng[i] = same_nr * same_ng;
                    } else {
                        Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[%1%]", i), "vlam", "det");
                        RootMuller rootdigger(*this, [&](const dcomplex& x){return this->detS1(2. - 4e3*M_PI / x / k0, nrCache[i], ngCache[i]);}, log_stripe, stripe_root);
                        dcomplex start = (vlam == 0.)? 2e3*M_PI / k0 : vlam;
                        veffs[i] = freqv(rootdigger(start-DLAM, start+DLAM));
                        computeStripeNNg(i, i==main_stripe);
                    }
                } catch (...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error != std::exception_ptr()) std::rethrow_exception(error);
        } else {
            // Compute effective frequencies for just one stripe
            writelog(LOG_DETAIL, "Computing effective frequency for vertical stripe %1%", rstripe);
#           ifndef NDEBUG
                std::stringstream nrgs;
                for (auto nr = nrCache[rstripe].end(), ng = ngCache[rstripe].end(); nr != nrCache[rstripe].begin();) {
                    --nr; --ng;
                    nrgs << ", (" << str(*nr) << ")/(" << str(*ng) << ")";
                }
                writelog(LOG_DEBUG, "Nr/Ng[%1%] = [%2% ]", rstripe, nrgs.str().substr(1));
#           endif
            Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[%1%]", rstripe), "vlam", "det");
            RootMuller rootdigger(*this,
                                  [&](const dcomplex& x){
                                      return this->detS1(2. - 4e3*M_PI / x / k0, nrCache[rstripe], ngCache[rstripe]);
                                  },
                                  log_stripe,
                                  stripe_root
                                 );
            dcomplex start = (vlam == 0.)? 2e3*M_PI / k0 : vlam;
            veffs[rstripe] = freqv(rootdigger(start-DLAM, start+DLAM));
            // Compute veffs and neffs for other stripes
            computeStripeNNg(rstripe, true);
            for (size_t i = 0; i < rsize; ++i)
                if (i != rstripe) computeStripeNNg(i);
        }
        assert(zintegrals.size() == zsize);

#       ifndef NDEBUG
            std::stringstream strv; for (size_t i = 0; i < veffs.size(); ++i) strv << ", " << str(veffs[i]);
            writelog(LOG_DEBUG, "stripes veffs = [%1% ]", strv.str().substr(1));
            std::stringstream strn; for (size_t i = 0; i < nng.size(); ++i) strn << ", " << str(nng[i]);
            writelog(LOG_DEBUG, "stripes <nng> = [%1% ]", strn.str().substr(1));
#       endif

        have_veffs = true;

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
        phas = exp(I * kz[zbegin] * (mesh->axis1->at(zbegin)-mesh->axis1->at(zbegin-1)));

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
        if (i != mesh->axis1->size())
            phas = exp(I * kz[i] * (mesh->axis1->at(i)-mesh->axis1->at(i-1)));

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
        dcomplex f;
        if (emission == TOP) {
            f = 1. / (*saveto)[zsize-1].F / sqrt(NG[zsize-1]);
            (*saveto)[zsize-1] = FieldZ(1., 0.);
        } else {
            // we dont have to scale fields as we have already set (*saveto)[zbegin] = FieldZ(0., 1.)
            f = 1. / sqrt(NG[zbegin]);
            (*saveto)[zsize-1].B = 0.;
        }
        for (size_t i = zbegin; i < zsize-1; ++i) (*saveto)[i] *= f;
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


void EffectiveFrequencyCylSolver::computeStripeNNg(size_t stripe, bool save_integrals)
{
    size_t stripe0 = (rstripe < 0)? stripe : rstripe;

    nng[stripe] = 0.;

    if (stripe != stripe0) veffs[stripe] = 0.;

    std::vector<FieldZ> zfield(zsize);

    dcomplex veff = veffs[stripe0];

    // Compute fields
    detS1(veff, nrCache[stripe0], ngCache[stripe0], &zfield);

    double sum = 0.;

    if (save_integrals) {
        #pragma omp critical
        zintegrals.resize(zsize);
    }

    for (size_t i = zbegin+1; i < zsize-1; ++i) {
        double d = mesh->axis1->at(i) - mesh->axis1->at(i-1);
        double weight = 0.;
        dcomplex kz = k0 * sqrt(nrCache[stripe0][i]*nrCache[stripe0][i] - veff * nrCache[stripe0][i]*ngCache[stripe0][i]);
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
            weight = -imag(zfield[i].F * conj(zfield[i].F) * w_ff +
                           zfield[i].F * conj(zfield[i].B) * w_fb +
                           zfield[i].B * conj(zfield[i].F) * w_bf +
                           zfield[i].B * conj(zfield[i].B) * w_bb);
        }
        sum += weight;
        if (save_integrals) {
            #pragma omp critical
            zintegrals[i] = weight;
        }
        nng[stripe] += weight * nrCache[stripe][i] * ngCache[stripe][i];
        if (stripe != stripe0) {
            veffs[stripe] += weight * (nrCache[stripe][i]*nrCache[stripe][i] - nrCache[stripe0][i]*nrCache[stripe0][i]);
        }
    }

    if (stripe != stripe0) {
        veffs[stripe] += veff * nng[stripe0] * sum;
        veffs[stripe] /= nng[stripe];
    }

    nng[stripe] /= sum;
}

double EffectiveFrequencyCylSolver::integrateBessel(Mode& mode)
{
    double sum = 0;
    for (size_t i = 0; i != rsize; ++i) {
        double start = mesh->axis0->at(i);
        double end = (i != rsize-1)? mesh->axis0->at(i+1) : 3.0 * mesh->axis0->at(mesh->axis0->size()-1);
        double err = perr;
        mode.rweights[i] = patterson<double>([this,&mode](double r){return r * abs2(mode.rField(r));}, start, end, err);
        //TODO use exponential asymptotic approximation to compute weight in the last stripe
        sum += mode.rweights[i];
    }
    //TODO consider m <> 0
    double f = 1e12 / sum; for (double& w: mode.rweights) w *= f;
    return 2.*M_PI * sum;
}

dcomplex EffectiveFrequencyCylSolver::detS(const dcomplex& lam, plask::solvers::effective::EffectiveFrequencyCylSolver::Mode& mode, bool save)
{
    dcomplex v = freqv(lam);

    // In the outermost layer, there is only an outgoing wave, so the solution is only the Hankel function
    mode.rfields[rsize-1] = FieldR(0., 1.);

    for (size_t i = rsize-1; i > 0; --i) {

        double r = mesh->axis0->at(i);

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
        if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)\n @ r = %3% um, lam = %4% nm, vlam = %6% nm",
            mode.m, str(x1), r, str(lambda(v)), str(lambda(veffs[i-1])));
        zbesh(x1.real(), x1.imag(), mode.m, 1, MH, 2, Hr, Hi, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)\n @ r = %3% um, lam = %4% nm, vlam = %6% nm",
            mode.m, str(x1), r, str(lambda(v)), str(lambda(veffs[i-1])));
        for (int j = 0; j < 2; ++j) { J1[j] = dcomplex(Jr[j], Ji[j]); H1[j] = dcomplex(Hr[j], Hi[j]); }

        zbesj(x2.real(), x2.imag(), mode.m, 1, 2, Jr, Ji, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute J(%1%, %2%)\n @ r = %3% um, lam = %4% nm, vlam = %6% nm",
            mode.m, str(x1), r, str(lambda(v)), str(lambda(veffs[i])));
        zbesh(x2.real(), x2.imag(), mode.m, 1, MH, 2, Hr, Hi, nz, ierr);
        if (ierr != 0) throw ComputationError(getId(), "Could not compute H(%1%, %2%)\n @ r = %3% um, lam = %4% nm, vlam = %6% nm",
            mode.m, str(x1), r, str(lambda(v)), str(lambda(veffs[i])));
        for (int j = 0; j < 2; ++j) { J2[j] = dcomplex(Jr[j], Ji[j]); H2[j] = dcomplex(Hr[j], Hi[j]); }

        MatrixR A(       J1[0],                   H1[0],
                  mode.m*J1[0] - x1*J1[1], mode.m*H1[0] - x1*H1[1]);

        MatrixR B(       J2[0],                   H2[0],
                  mode.m*J2[0] - x2*J2[1], mode.m*H2[0] - x2*H2[1]);

        mode.rfields[i-1] = A.solve(B * mode.rfields[i]);
    }

    if (save) {
        dcomplex f = 1e6 * sqrt(1. / integrateBessel(mode)); // 1e6: V/µm -> V/m
        for (size_t r = 0; r != rsize; ++r) mode.rfields[r] *= f;
    }

    // In the innermost area there must not be any infinity, so H = 0.
    //return rfields[0].H / rfields[0].J; // to stress the difference between J and H
    return mode.rfields[0].H;
}


double EffectiveFrequencyCylSolver::getTotalAbsorption(const Mode& mode)
{
    double result = 0.;

    for (size_t ir = 0; ir < rsize; ++ir) {
        for (size_t iz = zbegin+1; iz < zsize-1; ++iz) {
            double absp = - 2. * real(nrCache[ir][iz]) * imag(nrCache[ir][iz]);
            result += absp * mode.rweights[ir] * zintegrals[iz]; // [dV] = µm³
            // double err = 1e-6, erz = 1e-6;
            // double rstart = mesh->axis0[ir];
            // double rend = (ir != rsize-1)? mesh->axis0[ir+1] : 3.0 * mesh->axis0[ir];
            // result += absp * 2.*M_PI *
            //     patterson<double>([this,&mode](double r){return r * abs2(mode.rField(r));}, rstart,  rend, err) *
            //     patterson<double>([&](double z){
            //         size_t stripe = getMainStripe();
            //         dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - veffs[stripe] * nrCache[stripe][iz]*ngCache[stripe][iz]);
            //         if (real(kz) < 0.) kz = -kz;
            //         z -= mesh->axis1[iz];
            //         dcomplex phasz = exp(- I * kz * z);
            //         return abs2(zfields[iz].F * phasz + zfields[iz].B / phasz);
            //     }, mesh->axis1[iz-1], mesh->axis1[iz], erz);
        }
    }
    result *= 2e-9 * M_PI / real(mode.lam) * mode.power; // 1e-9: µm³ / nm -> m², 2: ½ is already hidden in mode.power
    return result;
}

double EffectiveFrequencyCylSolver::getTotalAbsorption(size_t num)
{
    if (modes.size() <= num || k0 != old_k0) throw NoValue("absorption");

    updateCache();

    if (!modes[num].have_fields) {
        size_t stripe = getMainStripe();
        detS1(veffs[stripe], nrCache[stripe], ngCache[stripe], &zfields); // compute vertical part
        detS(modes[num].lam, modes[num], true); // compute horizontal part
        modes[num].have_fields = true;
    }

    return getTotalAbsorption(modes[num]);
}


double EffectiveFrequencyCylSolver::getGainIntegral(const Mode& mode)
{
    double result = 0.;

    auto midmesh = mesh->getMidpointsMesh();

    for (size_t ir = 0; ir < rsize; ++ir) {
        for (size_t iz = zbegin+1; iz < zsize-1; ++iz) {
            auto roles = geometry->getRolesAt(midmesh->at(ir, iz-1));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                double absp = - 2. * real(nrCache[ir][iz]) * imag(nrCache[ir][iz]);
                result += absp * mode.rweights[ir] * zintegrals[iz]; // [dV] = µm³
            }
        }
    }
    result *= 2e-9 * M_PI / real(mode.lam) * mode.power; // 1e-9: µm³ / nm -> m², 2: ½ is already hidden in mode.power
    return -result;
}

double EffectiveFrequencyCylSolver::getGainIntegral(size_t num)
{
    if (modes.size() <= num || k0 != old_k0) throw NoValue("absorption");

    updateCache();

    if (!modes[num].have_fields) {
        size_t stripe = getMainStripe();
        detS1(veffs[stripe], nrCache[stripe], ngCache[stripe], &zfields); // compute vertical part
        detS(modes[num].lam, modes[num], true); // compute horizontal part
        modes[num].have_fields = true;
    }

    return getGainIntegral(modes[num]);
}


plask::DataVector<const double> EffectiveFrequencyCylSolver::getLightMagnitude(int num, const MeshD<2>& dst_mesh, InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (modes.size() <= num || k0 != old_k0) throw NoValue(LightMagnitude::NAME);

    size_t stripe = getMainStripe();

    if (!modes[num].have_fields) {
        // Compute vertical part
        detS1(veffs[stripe], nrCache[stripe], ngCache[stripe], &zfields);
        // Compute horizontal part
        detS(modes[num].lam, modes[num], true);
        #ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = 0; i < rsize; ++i)
                nrs << "), (" << str(modes[num].rfields[i].J) << ":" << str(modes[num].rfields[i].H);
            writelog(LOG_DEBUG, "horizontal fields = [%1%) ]", nrs.str().substr(2));
        }
        #endif
        modes[num].have_fields = true;
    }

    DataVector<double> results(dst_mesh.size());

    if (!getLightMagnitude_Efficient(num, stripe, dst_mesh, results)) {
        std::exception_ptr error; // needed to handle exceptions from OMP loop

        const double power = 1e-3 * modes[num].power; // 1e-3 mW->W

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

            size_t iz = mesh->axis1->findIndex(z);
            if (iz >= zsize) iz = zsize-1;
            else if (iz < zbegin) iz = zbegin;
            dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - veffs[stripe] * nrCache[stripe][iz]*ngCache[stripe][iz]);
            if (real(kz) < 0.) kz = -kz;
            z -= mesh->axis1->at(max(int(iz)-1, 0));
            dcomplex phasz = exp(- I * kz * z);
            val *= zfields[iz].F * phasz + zfields[iz].B / phasz;

            results[id] = power * abs2(val);
        }

        if (error) std::rethrow_exception(error);
    }

    return results;
}

bool EffectiveFrequencyCylSolver::getLightMagnitude_Efficient(size_t num, size_t stripe, const MeshD<2>& dst_mesh, DataVector<double>& results)
{
    if (auto rect_mesh_ptr = dynamic_cast<const RectangularMesh<2>*>(&dst_mesh)) {

        const RectangularMesh<2>& rect_mesh = *rect_mesh_ptr;

        std::vector<dcomplex> valr(rect_mesh.axis0->size());
        std::vector<dcomplex> valz(rect_mesh.axis1->size());

        std::exception_ptr error; // needed to handle exceptions from OMP loop

        const double power = 1e-3 * modes[num].power; // 1e-3 mW->W

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (size_t idr = 0; idr < rect_mesh.axis0->size(); ++idr) {
                if (error) continue;
                double r = rect_mesh.axis0->at(idr);
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
                for (size_t idz = 0; idz < rect_mesh.axis1->size(); ++idz) {
                    double z = rect_mesh.axis1->at(idz);
                    size_t iz = mesh->axis1->findIndex(z);
                    if (iz >= zsize) iz = zsize-1;
                    else if (iz < zbegin) iz = zbegin;
                    dcomplex kz = k0 * sqrt(nrCache[stripe][iz]*nrCache[stripe][iz] - veffs[stripe] * nrCache[stripe][iz]*ngCache[stripe][iz]);
                    if (real(kz) < 0.) kz = -kz;
                    z -= mesh->axis1->at(max(int(iz)-1, 0));
                    dcomplex phasz = exp(- I * kz * z);
                    valz[idz] = zfields[iz].F * phasz + zfields[iz].B / phasz;
                }

                if (rect_mesh.getIterationOrder() == RectangularMesh<2>::ORDER_10) {
                    #pragma omp for
                    for (size_t i1 = 0; i1 < rect_mesh.axis1->size(); ++i1) {
                        double* data = results.data() + i1 * rect_mesh.axis0->size();
                        for (size_t i0 = 0; i0 < rect_mesh.axis0->size(); ++i0) {
                            dcomplex f = valr[i0] * valz[i1];
                            data[i0] = power * abs2(f);
                        }
                    }
                } else {
                    #pragma omp for
                    for (size_t i0 = 0; i0 < rect_mesh.axis0->size(); ++i0) {
                        double* data = results.data() + i0 * rect_mesh.axis1->size();
                        for (size_t i1 = 0; i1 < rect_mesh.axis1->size(); ++i1) {
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


DataVector<const Tensor3<dcomplex>> EffectiveFrequencyCylSolver::getRefractiveIndex(const MeshD<2>& dst_mesh, double lam, InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting refractive indices");
    dcomplex ok0 = k0;
    if (!isnan(lam) && lam != 0.) k0 = 2e3*M_PI / lam;
    try { updateCache(); }
    catch(...) { k0 = ok0; throw; }
    k0 = ok0;
    auto target_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    DataVector<Tensor3<dcomplex>> result(dst_mesh.size());
    for (size_t j = 0; j != dst_mesh.size(); ++j) {
        auto point = target_mesh[j];
        size_t ir = this->mesh->axis0->findIndex(point.c0); if (ir != 0) --ir; if (ir >= rsize) ir = rsize-1;
        size_t iz = this->mesh->axis1->findIndex(point.c1); if (iz < zbegin) iz = zbegin; else if (iz >= zsize) iz = zsize-1;
        result[j] = Tensor3<dcomplex>(nrCache[ir][iz]);
    }
    return result;
}


DataVector<const double> EffectiveFrequencyCylSolver::getHeat(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    // This is somehow naive implementation using the field value from the mesh points. The heat may be slightly off
    // in case of fast varying light intensity and too sparse mesh.

    writelog(LOG_DETAIL, "Getting heat absorbed from %1% mode%2%", modes.size(), (modes.size()==1)? "" : "s");

    DataVector<double> result(dst_mesh.size(), 0.);

    if (modes.size() == 0) return result;

    updateCache();

    for (size_t m = 0; m != modes.size(); ++m) { // we sum heats from all modes
        result += 2e9*M_PI / real(modes[m].lam) * getLightMagnitude(m, dst_mesh, method); // 1e9: 1/nm -> 1/m
    }
    auto mat_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    for (size_t j = 0; j != result.size(); ++j) {
        auto point = mat_mesh[j];
        size_t ir = this->mesh->axis0->findIndex(point.c0); if (ir != 0) --ir; if (ir >= rsize) ir = rsize-1;
        size_t iz = this->mesh->axis1->findIndex(point.c1); if (iz < zbegin) iz = zbegin; else if (iz >= zsize) iz = zsize-1;
        double absp = - 2. * real(nrCache[ir][iz]) * imag(nrCache[ir][iz]);
        result[j] *= absp;
    }
    return result;
}


}}} // namespace plask::solvers::effective
