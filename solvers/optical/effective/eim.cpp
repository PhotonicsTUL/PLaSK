#include <exception>
#include "eim.h"

using plask::dcomplex;

#define DNEFF 1e-9

namespace plask { namespace solvers { namespace effective {

EffectiveIndex2DSolver::EffectiveIndex2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectangularMesh<2>>(name),
    log_value(dataLog<dcomplex, dcomplex>("Neff", "Neff", "det")),
    recompute_neffs(true),
    stripex(0.),
    polarization(TE),
    vneff(0.),
    outdist(0.1),
    outNeff(this, &EffectiveIndex2DSolver::getEffectiveIndex, &EffectiveIndex2DSolver::nmodes),
    outLightIntensity(this, &EffectiveIndex2DSolver::getLightIntenisty, &EffectiveIndex2DSolver::nmodes),
    outRefractiveIndex(this, &EffectiveIndex2DSolver::getRefractiveIndex),
    outHeat(this, &EffectiveIndex2DSolver::getHeat),
    k0(2e3*M_PI/980) {
    inTemperature = 300.;
    inGain = NAN;
    root.tolx = 1.0e-6;
    root.tolf_min = 1.0e-8;
    root.tolf_max = 1.0e-6;
    root.maxiter = 500;
    stripe_root.tolx = 1.0e-8;
    stripe_root.tolf_min = 1.0e-8;
    stripe_root.tolf_max = 1.0e-6;
    stripe_root.maxiter = 500;
    inTemperature.changedConnectMethod(this, &EffectiveIndex2DSolver::onInputChange);
    inGain.changedConnectMethod(this, &EffectiveIndex2DSolver::onInputChange);
}


void EffectiveIndex2DSolver::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "mode") {
            auto pol = reader.getAttribute("polarization");
            if (pol) {
                if (*pol == "TE") polarization = TE;
                else if (*pol == "TM") polarization = TM;
                else throw BadInput(getId(), "Wrong polarization specification '%1%' in XML", *pol);
            }
            // auto sym = reader.getAttribute("symmetry");
            // if (sym) {
            //     if (*sym == "0" || *sym == "none" ) {
            //         symmetry = SYMMETRY_NONE;
            //     }
            //     else if (*sym == "positive" || *sym == "pos" || *sym == "symmeric" || *sym == "+" || *sym == "+1") {
            //         symmetry = SYMMETRY_POSITIVE;
            //     }
            //     else if (*sym == "negative" || *sym == "neg" || *sym == "anti-symmeric" || *sym == "antisymmeric" || *sym == "-" || *sym == "-1") {
            //         symmetry = SYMMETRY_NEGATIVE;
            //     } else throw BadInput(getId(), "Wrong symmetry specification '%1%' in XML", *sym);
            // }
            k0 = 2e3*M_PI / reader.getAttribute<double>("wavelength",  real(2e3*M_PI / k0));
            stripex = reader.getAttribute<double>("vat", stripex);
            vneff = reader.getAttribute<dcomplex>("vneff", vneff);
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
        } else if (param == "mirrors") {
            double R1 = reader.requireAttribute<double>("R1");
            double R2 = reader.requireAttribute<double>("R2");
            mirrors.reset(std::make_pair(R1,R2));
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


std::vector<dcomplex> EffectiveIndex2DSolver::searchVNeffs(dcomplex neff1, dcomplex neff2, size_t resteps, size_t imsteps, dcomplex eps)
{
    updateCache();

    size_t stripe = mesh->tran().findIndex(stripex);
    if (stripe < xbegin) stripe = xbegin;
    else if (stripe >= xend) stripe = xend-1;

    if (eps.imag() == 0.) eps.imag(eps.real());

    if (real(eps) <= 0. || imag(eps) <= 0.)
        throw BadInput(this->getId(), "Bad precision specified");

    double re0 = real(neff1), im0 = imag(neff1);
    double re1 = real(neff2), im1 = imag(neff2);
    if (re0 > re1) std::swap(re0, re1);
    if (im0 > im1) std::swap(im0, im1);

    if (re0 == 0. && re1 == 0.) {
        re0 = 1e30;
        re1 = -1e30;
        for (size_t i = ybegin; i != yend; ++i) {
            dcomplex n = nrCache[stripe][i];
            if (n.real() < re0) re0 = n.real();
            if (n.real() > re1) re1 = n.real();
        }
    } else if (re0 == 0. || re1 == 0.)
        throw BadInput(getId(), "Bad area to browse specified");
    if (im0 == 0. && im1 == 0.) {
        im0 = 1e30;
        im1 = -1e30;
        for (size_t i = ybegin; i != yend; ++i) {
            dcomplex n = nrCache[stripe][i];
            if (n.imag() < im0) im0 = n.imag();
            if (n.imag() > im1) im1 = n.imag();
        }
    }
    neff1 = dcomplex(re0,im0);
    neff2 = dcomplex(re1,im1);

    auto ranges = findZeros(this, [&](const dcomplex& z){return this->detS1(z,nrCache[stripe]);}, neff1, neff2, resteps, imsteps, eps);
    std::vector<dcomplex> results; results.reserve(ranges.size());
    for (auto zz: ranges) results.push_back(0.5 * (zz.first+zz.second));

    if (maxLoglevel >= LOG_RESULT) {
        if (results.size() != 0) {
            Data2DLog<dcomplex,dcomplex> logger(getId(), format("stripe[%1%]", stripe-xbegin), "neff", "det");
            std::string msg = "Found vertical effective indices at: ";
            for (auto z: results) {
                msg += str(z) + ", ";
                logger(z, detS1(z,nrCache[stripe]));
            }
            writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
        } else
            writelog(LOG_RESULT, "Did not find any vertical effective indices");
    }

    return results;
}


size_t EffectiveIndex2DSolver::findMode(dcomplex neff, Symmetry symmetry)
{
    writelog(LOG_INFO, "Searching for the mode starting from Neff = %1%", str(neff));
    stageOne();
    Mode mode(this, symmetry);
    mode.neff = RootMuller(*this, [this,&mode](const dcomplex& x){return this->detS(x,mode);}, log_value, root)(neff-DNEFF, neff+DNEFF);
    return insertMode(mode);
}


size_t EffectiveIndex2DSolver::findMode(dcomplex neff1, dcomplex neff2, Symmetry symmetry)
{
    writelog(LOG_INFO, "Searching for the mode between Neffs %1% and %2%", str(neff1), str(neff2));
    stageOne();
    Mode mode(this, symmetry);
    mode.neff = RootMuller(*this, [this,&mode](const dcomplex& x){return this->detS(x,mode);}, log_value, root)(neff1, neff2);
    return insertMode(mode);
}


std::vector<size_t> EffectiveIndex2DSolver::findModes(dcomplex neff1, dcomplex neff2, Symmetry symmetry, size_t resteps, size_t imsteps, dcomplex eps)
{
    stageOne();

    if (eps.imag() == 0.) eps.imag(eps.real());

    if (real(eps) <= 0. || imag(eps) <= 0.)
        throw BadInput(this->getId(), "Bad precision specified");

    double re0 = real(neff1), im0 = imag(neff1);
    double re1 = real(neff2), im1 = imag(neff2);
    if (re0 > re1) std::swap(re0, re1);
    if (im0 > im1) std::swap(im0, im1);

    if (re0 == 0. && re1 == 0.) {
        re0 = 1e30;
        re1 = -1e30;
        for (size_t i = xbegin; i != xend; ++i) {
            dcomplex n = sqrt(epsilons[i]);
            if (n.real() < re0) re0 = n.real();
            if (n.real() > re1) re1 = n.real();
        }
    } else if (re0 == 0. || re1 == 0.)
        throw BadInput(getId(), "Bad area to browse specified");
    if (im0 == 0. && im1 == 0.) {
        im0 = 1e30;
        im1 = -1e30;
        for (size_t i = xbegin; i != xend; ++i) {
            dcomplex n = sqrt(epsilons[i]);
            if (n.imag() < im0) im0 = n.imag();
            if (n.imag() > im1) im1 = n.imag();
        }
    }
    neff1 = dcomplex(re0,im0);
    neff2 = dcomplex(re1,im1);

    Mode mode(this, symmetry);
    auto results = findZeros(this, [this,&mode](dcomplex z){return this->detS(z,mode);}, neff1, neff2, resteps, imsteps, eps);

    std::vector<size_t> idx(results.size());

    if (results.size() != 0) {
        Data2DLog<dcomplex,dcomplex> logger(getId(), "Neffs", "Neff", "det");
        RootMuller refine(*this, [this,&mode](const dcomplex& v){return this->detS(v,mode);}, logger, root);
        std::string msg = "Found modes at: ";
        for (auto zz: results) {
            dcomplex z;
            try {
                z = refine(zz.first, zz.second);
            } catch (ComputationError) {
                continue;
            }
            mode.neff = z;
            idx.push_back(insertMode(mode));
            msg += str(z) + ", ";
        }
        writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
    } else
        writelog(LOG_RESULT, "Did not find any modes");

    return idx;
}


size_t EffectiveIndex2DSolver::setMode(dcomplex neff, Symmetry sym)
{
    if (!initialized) {
        writelog(LOG_WARNING, "Solver invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    Mode mode(this, sym);
    mode.neff = neff;
    double det = abs(detS(neff, mode));
    if (det > root.tolf_max)
        writelog(LOG_WARNING, "Provided effective index does not correspond to any mode (det = %1%)", det);
    writelog(LOG_INFO, "Setting mode at %1%", str(neff));
    return insertMode(mode);
}

void EffectiveIndex2DSolver::onInitialize()
{
    if (!geometry) throw NoGeometryException(getId());

    // Set default mesh
    if (!mesh) setSimpleMesh();

    xbegin = 0;
    ybegin = 0;
    xend = mesh->axis0->size() + 1;
    yend = mesh->axis1->size() + 1;

    if (geometry->isExtended(Geometry::DIRECTION_TRAN, false) &&
        abs(mesh->axis0->at(0) - geometry->getChild()->getBoundingBox().lower.c0) < SMALL)
        xbegin = 1;
    if (geometry->isExtended(Geometry::DIRECTION_VERT, false) &&
        abs(mesh->axis1->at(0) - geometry->getChild()->getBoundingBox().lower.c1) < SMALL)
        ybegin = 1;
    if (geometry->isExtended(Geometry::DIRECTION_TRAN, true) &&
        abs(mesh->axis0->at(mesh->axis0->size()-1) - geometry->getChild()->getBoundingBox().upper.c0) < SMALL)
        --xend;
    if (geometry->isExtended(Geometry::DIRECTION_VERT, true) &&
        abs(mesh->axis1->at(mesh->axis1->size()-1) - geometry->getChild()->getBoundingBox().upper.c1) < SMALL)
        --yend;

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(xend, std::vector<dcomplex,aligned_allocator<dcomplex>>(yend));
    epsilons.resize(xend);

    yfields.resize(yend);

    need_gain = false;
    recompute_neffs = true;
}


void EffectiveIndex2DSolver::onInvalidate()
{
    if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
    modes.clear();
    outNeff.fireChanged();
    outLightIntensity.fireChanged();
}

/********* Here are the computations *********/


void EffectiveIndex2DSolver::updateCache()
{
    bool fresh = !initCalculation();

    if (fresh || inTemperature.changed() || (need_gain && inGain.changed()) || recompute_neffs) {
        // we need to update something

        if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
            if (fresh) // Make sure we have only positive points
                for (auto x: *mesh->axis0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
            if (mesh->axis0->at(0) == 0.) xbegin = 1;
        }

        if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        modes.clear();

        double w = real(2e3*M_PI / k0);

        shared_ptr<RectilinearAxis> axis0, axis1;
        {
            shared_ptr<RectangularMesh<2>> midmesh = mesh->getMidpointsMesh();
            axis0 = make_shared<RectilinearAxis>(*midmesh->axis0);
            axis1 = make_shared<RectilinearAxis>(*midmesh->axis1);
        }

        if (xbegin == 0) {
            if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) axis0->addPoint(0.5 * mesh->axis0->at(0));
            else axis0->addPoint(mesh->axis0->at(0) - outdist);
        }
        if (xend == mesh->axis0->size()+1)
            axis0->addPoint(mesh->axis0->at(mesh->axis0->size()-1) + outdist);
        if (ybegin == 0)
            axis1->addPoint(mesh->axis1->at(0) - outdist);
        if (yend == mesh->axis1->size()+1)
            axis1->addPoint(mesh->axis1->at(mesh->axis1->size()-1) + outdist);

        writelog(LOG_DEBUG, "Updating refractive indices cache");
        RectangularMesh<2> midmesh(axis0, axis1, mesh->getIterationOrder());
        auto temp = inTemperature(midmesh);
        bool have_gain = false;
        DataVector<const double> gain;

        for (size_t ix = xbegin; ix < xend; ++ix) {
            for (size_t iy = ybegin; iy < yend; ++iy) {
                size_t idx = midmesh.index(ix-xbegin, iy-ybegin);
                double T = temp[idx];
                auto point = midmesh[idx];
                auto roles = geometry->getRolesAt(point);
                if (roles.find("QW") == roles.end() && roles.find("QD") == roles.end() && roles.find("gain") == roles.end())
                    nrCache[ix][iy] = geometry->getMaterial(point)->Nr(w, T);
                else {  // we ignore the material absorption as it should be considered in the gain already
                    need_gain = true;
                    if (!have_gain) {
                        gain = inGain(midmesh, w);
                        have_gain = false;
                    }
                    double g = gain[idx];
                    nrCache[ix][iy] = dcomplex( real(geometry->getMaterial(point)->Nr(w, T)),
                                                w * g * 7.95774715459e-09 );
                }
            }
        }
        recompute_neffs = true;
    }
}


void EffectiveIndex2DSolver::stageOne()
{
    updateCache();

    if (recompute_neffs) {

        // Compute effective index of the main stripe
        size_t stripe = mesh->tran().findIndex(stripex);
        if (stripe < xbegin) stripe = xbegin;
        else if (stripe >= xend) stripe = xend-1;
        writelog(LOG_DETAIL, "Computing effective index for vertical stripe %1% (polarization %2%)", stripe-xbegin, (polarization==TE)?"TE":"TM");
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (ptrdiff_t j = yend-1; j >= ptrdiff_t(ybegin); --j) nrs << ", " << str(nrCache[stripe][j]);
//             writelog(LOG_DEBUG, "Nr[%1%] = [%2% ]", stripe-xbegin, nrs.str().substr(1));
//         }
// #endif
        Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[%1%]", stripe-xbegin), "neff", "det");
        RootMuller rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[stripe]);}, log_stripe, stripe_root);
        if (vneff == 0.) {
            dcomplex maxn = *std::max_element(nrCache[stripe].begin(), nrCache[stripe].end(),
                                              [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
            vneff = 0.999 * real(maxn);
        }
        vneff = rootdigger(vneff-DNEFF, vneff+DNEFF);

        // Compute field weights
        computeWeights(stripe);

        // Compute effective indices
        for (size_t i = xbegin; i < xend; ++i) {
            epsilons[i] = vneff*vneff;
            for (size_t j = ybegin; j < yend; ++j) {
                epsilons[i] += weights[j] * (nrCache[i][j]*nrCache[i][j] - nrCache[stripe][j]*nrCache[stripe][j]);
            }
        }
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i) {
                dcomplex n = sqrt(epsilons[i]); if (abs(n.real()) < 1e-10) n.real(0.); if (abs(n.imag()) < 1e-10) n.imag(0.);
                nrs << ", " << str(n);
            }
            writelog(LOG_DEBUG, "horizontal neffs = [%1% ]", nrs.str().substr(1));
        }
#endif
        recompute_neffs = false;

        double rmin=INFINITY, rmax=-INFINITY, imin=INFINITY, imax=-INFINITY;
        for (size_t i = xbegin; i < xend; ++i) {
            dcomplex n = sqrt(epsilons[i]);
            if (real(n) < rmin) rmin = real(n);
            if (real(n) > rmax) rmax = real(n);
            if (imag(n) < imin) imin = imag(n);
            if (imag(n) > imax) imax = imag(n);
        }
        writelog(LOG_DETAIL, "Effective index should be between %1% and %2%", str(dcomplex(rmin,imin)), str(dcomplex(rmax,imax)));
    }
}

dcomplex EffectiveIndex2DSolver::detS1(const plask::dcomplex& x, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR, bool save)
{
    if (save) yfields[ybegin] = Field(0., 1.);

    std::vector<dcomplex,aligned_allocator<dcomplex>> ky(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        ky[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
        if (imag(ky[i]) > 0.) ky[i] = -ky[i];
    }

    dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S

    dcomplex phas = 1.;
    if (ybegin != 0)
        phas = exp(I * ky[ybegin] * (mesh->axis1->at(ybegin)-mesh->axis1->at(ybegin-1)));

    for (size_t i = ybegin+1; i < yend; ++i) {
        // Compute shift inside one layer
        s1 *= phas;
        s3 *= phas * phas;
        s4 *= phas;
        // Compute matrix after boundary
        dcomplex f = (polarization==TM)? NR[i-1]/NR[i] : 1.;
        dcomplex p = 0.5 + 0.5 * ky[i] / ky[i-1] * f*f;
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
            phas = exp(I * ky[i] * (mesh->axis1->at(i)-mesh->axis1->at(i-1)));

        // Compute fields
        if (save) {
            dcomplex F = -s2/s1, B = (s1*s4-s2*s3)/s1;    // Assume  F0 = 0  B0 = 1
            double aF = abs(F), aB = abs(B);
            // zero very small fields to avoid errors in plotting for long layers
            if (aF < 1e-8 * aB) F = 0.;
            if (aB < 1e-8 * aF) B = 0.;
            yfields[i] = Field(F, B);
        }
    }

    if (save) yfields[yend-1].B = 0.;
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i)
//                 nrs << "), (" << str(yfields[i].F) << ":" << str(yfields[i].B);
//             writelog(LOG_DEBUG, "vertical fields = [%1%) ]", nrs.str().substr(2));
//         }
// #endif

    return s1*s4 - s2*s3;
}


void EffectiveIndex2DSolver::computeWeights(size_t stripe)
{
    // Compute fields
    detS1(vneff, nrCache[stripe], true);

    weights.resize(yend);
    {
        double ky = abs(imag(k0 * sqrt(nrCache[stripe][ybegin]*nrCache[stripe][ybegin] - vneff*vneff)));
        weights[ybegin] = abs2(yfields[ybegin].B) * 0.5 / ky;
    }
    {
        double ky = abs(imag(k0 * sqrt(nrCache[stripe][yend-1]*nrCache[stripe][yend-1] - vneff*vneff)));
        weights[yend-1] = abs2(yfields[yend-1].F) * 0.5 / ky;
    }
    double sum = weights[ybegin] + weights[yend-1];

    for (size_t i = ybegin+1; i < yend-1; ++i) {
        double d = mesh->axis1->at(i)-mesh->axis1->at(i-1);
        dcomplex ky = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - vneff*vneff); if (imag(ky) > 0.) ky = -ky;
        dcomplex w_ff, w_bb, w_fb, w_bf;
        if (d != 0.) {
            if (abs(imag(ky)) > SMALL) {
                dcomplex kk = ky - conj(ky);
                w_ff =   (exp(-I*d*kk) - 1.) / kk;
                w_bb = - (exp(+I*d*kk) - 1.) / kk;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            if (abs(real(ky)) > SMALL) {
                dcomplex kk = ky + conj(ky);
                w_fb =   (exp(-I*d*kk) - 1.) / kk;
                w_bf = - (exp(+I*d*kk) - 1.) / kk;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            dcomplex weight = yfields[i].F * conj(yfields[i].F) * w_ff +
                              yfields[i].F * conj(yfields[i].B) * w_fb +
                              yfields[i].B * conj(yfields[i].F) * w_bf +
                              yfields[i].B * conj(yfields[i].B) * w_bb;
            weights[i] = -imag(weight);
        } else
            weights[i] = 0.;
        sum += weights[i];
    }

    sum = 1. / sum;
    double fact = sqrt(sum);
    for (size_t i = ybegin; i < yend; ++i) {
        weights[i] *= sum;
        yfields[i] *= fact;
    }
// #ifndef NDEBUG
//     {
//         std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i) nrs << ", " << str(weights[i]);
//         writelog(LOG_DEBUG, "vertical weights = [%1%) ]", nrs.str().substr(2));
//     }
// #endif
}

void EffectiveIndex2DSolver::normalizeFields(Mode& mode, const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx) {

    size_t start;

    double sum = abs2(mode.xfields[xend-1].F) * 0.5 / abs(imag(kx[xend-1]));
    if (mode.symmetry == SYMMETRY_NONE) {
        sum += abs2(mode.xfields[xbegin].B) * 0.5 / abs(imag(kx[xbegin]));
        start = xbegin+1;
    } else {
        start = xbegin;
    }

    for (size_t i = start; i < xend-1; ++i) {
        double d = mesh->axis0->at(i) - ((i == 0)? 0. : mesh->axis0->at(i-1));
        dcomplex w_ff, w_bb, w_fb, w_bf;
        if (d != 0.) {
            if (abs(imag(kx[i])) > SMALL) {
                dcomplex kk = kx[i] - conj(kx[i]);
                w_ff =   (exp(-I*d*kk) - 1.) / kk;
                w_bb = - (exp(+I*d*kk) - 1.) / kk;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            if (abs(real(kx[i])) > SMALL) {
                dcomplex kk = kx[i] + conj(kx[i]);
                w_fb =   (exp(-I*d*kk) - 1.) / kk;
                w_bf = - (exp(+I*d*kk) - 1.) / kk;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            sum -= imag(mode.xfields[i].F * conj(mode.xfields[i].F) * w_ff +
                        mode.xfields[i].F * conj(mode.xfields[i].B) * w_fb +
                        mode.xfields[i].B * conj(mode.xfields[i].F) * w_bf +
                        mode.xfields[i].B * conj(mode.xfields[i].B) * w_bb
                       );
        }
    }
    if (mode.symmetry != SYMMETRY_NONE) sum *= 2.;

    // Consider loss on the mirror
    double R1, R2;
    if (mirrors) {
        std::tie(R1,R2) = *mirrors;
    } else {
        const double lambda = real(2e3*M_PI / k0);
        const double n = real(mode.neff);
        const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
                     n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
        R1 = abs((n-n1) / (n+n1));
        R2 = abs((n-n2) / (n+n2));
    }
    if (emission == FRONT) sum *= R1;
    else sum *= R2;

    dcomplex f = sqrt(1e12 / sum);  // 1e12 because intensity in W/m² and integral computed in µm

    for (size_t i = xbegin; i < xend; ++i) {
        mode.xfields[i] *= f;
    }
}

dcomplex EffectiveIndex2DSolver::detS(const dcomplex& x, EffectiveIndex2DSolver::Mode& mode, bool save)
{
    // Adjust for mirror losses
    dcomplex neff2 = dcomplex(real(x), imag(x)-getMirrorLosses(x)); neff2 *= neff2;

    std::vector<dcomplex,aligned_allocator<dcomplex>> kx(xend);
    for (size_t i = xbegin; i < xend; ++i) {
        kx[i] = k0 * sqrt(epsilons[i] - neff2);
        if (imag(kx[i]) > 0.) kx[i] = -kx[i];
    }

    aligned_unique_ptr<Matrix[]> matrices;
    if (save) matrices.reset(aligned_malloc<Matrix>(xend-1));

    Matrix T = Matrix::eye();
    for (size_t i = xbegin; i < xend-1; ++i) {
        double d;
        if (i != xbegin) d = mesh->axis0->at(i) - mesh->axis0->at(i-1);
        else if (mode.symmetry != SYMMETRY_NONE) d = mesh->axis0->at(i);     // we have symmetry, so beginning of the transfer matrix is at the axis
        else d = 0.;
        dcomplex phas = exp(- I * kx[i] * d);
        // Transfer through boundary
        dcomplex f = (polarization==TE)? (sqrt(epsilons[i+1]/epsilons[i])) : 1.;
        dcomplex n = 0.5 * kx[i]/kx[i+1] * f*f;
        Matrix T1 = Matrix( (0.5+n), (0.5-n),
                            (0.5-n), (0.5+n) );
        T1.ff *= phas; T1.fb /= phas;
        T1.bf *= phas; T1.bb /= phas;
        if (save) matrices[i] = T1;
        T = T1 * T;
    }

    if (save) {
        mode.neff = x;

        mode.xfields[xend-1] = Field(1., 0.);
        for (size_t i = xend-1; i != xbegin; --i) {
            mode.xfields[i-1] = matrices[i-1].solve(mode.xfields[i]);
        }
        normalizeFields(mode, kx);
        mode.have_fields = true;
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i)
                nrs << "), (" << str(mode.xfields[i].F) << ":" << str(mode.xfields[i].B);
            writelog(LOG_DEBUG, "horizontal fields = [%1%) ]", nrs.str().substr(2));
        }
#endif
    }

    if (mode.symmetry == SYMMETRY_POSITIVE) return T.bf + T.bb;      // B0 = F0   Bn = 0
    else if (mode.symmetry == SYMMETRY_NEGATIVE) return T.bf - T.bb; // B0 = -F0  Bn = 0
    else return T.bb;                                                // F0 = 0    Bn = 0
}


plask::DataVector<const double> EffectiveIndex2DSolver::getLightIntenisty(int num, const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (outNeff.size() <= num) throw NoValue(LightIntensity::NAME);

    dcomplex neff = modes[num].neff;

    if (!modes[num].have_fields) detS(neff, modes[num], true);

    writelog(LOG_INFO, "Computing field distribution for Neff = %1%", str(neff));

    std::vector<dcomplex,aligned_allocator<dcomplex>> kx(xend);
    dcomplex neff2 = dcomplex(real(neff), imag(neff)-getMirrorLosses(real(neff))); neff2 *= neff2;
    for (size_t i = 0; i < xend; ++i) {
        kx[i] = k0 * sqrt(epsilons[i] - neff2);
        if (imag(kx[i]) > 0.) kx[i] = -kx[i];
    }

    size_t stripe = mesh->tran().findIndex(stripex);
    if (stripe < xbegin) stripe = xbegin;
    else if (stripe >= xend) stripe = xend-1;

    std::vector<dcomplex,aligned_allocator<dcomplex>> ky(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        ky[i] = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - vneff*vneff);
        if (imag(ky[i]) > 0.) ky[i] = -ky[i];
    }

    DataVector<double> results(dst_mesh.size());

    if (!getLightIntenisty_Efficient(num, dst_mesh, results, kx, ky)) {

        #pragma omp parallel for
        for (size_t idx = 0; idx < dst_mesh.size(); ++idx) {
            auto point = dst_mesh[idx];
            double x = point.tran();
            double y = point.vert();

            bool negate = false;
            if (x < 0. && modes[num].symmetry != SYMMETRY_NONE) {
                x = -x; if (modes[num].symmetry == SYMMETRY_NEGATIVE) negate = true;
            }
            size_t ix = mesh->tran().findIndex(x);
            if (ix >= xend) ix = xend-1;
            if (ix < xbegin) ix = xbegin;
            if (ix != 0) x -= mesh->tran()[ix-1];
            else if (modes[num].symmetry == SYMMETRY_NONE) x -= mesh->tran()[0];
            dcomplex phasx = exp(- I * kx[ix] * x);
            dcomplex val = modes[num].xfields[ix].F * phasx + modes[num].xfields[ix].B / phasx;
            if (negate) val = - val;

            size_t iy = mesh->vert().findIndex(y);
            if (iy >= yend) iy = yend-1;
            if (iy < ybegin) iy = ybegin;
            y -= mesh->vert()[max(int(iy)-1, 0)];
            dcomplex phasy = exp(- I * ky[iy] * y);
            val *= yfields[iy].F * phasy + yfields[iy].B / phasy;

            results[idx] = abs2(val);
        }
    }

    return results;
}

bool EffectiveIndex2DSolver::getLightIntenisty_Efficient(size_t num, const plask::MeshD<2>& dst_mesh, DataVector<double>& results,
                                                         const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx,
                                                         const std::vector<dcomplex,aligned_allocator<dcomplex>>& ky)
{
    if (const RectangularMesh<2>* rect_mesh_ptr = dynamic_cast<const RectangularMesh<2>*>(&dst_mesh)) {

        const RectangularMesh<2>& rect_mesh = *rect_mesh_ptr;

        std::vector<dcomplex,aligned_allocator<dcomplex>> valx(rect_mesh.tran().size());
        std::vector<dcomplex,aligned_allocator<dcomplex>> valy(rect_mesh.vert().size());

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (size_t idx = 0; idx < rect_mesh.tran().size(); ++idx) {
                double x = rect_mesh.tran()[idx];
                bool negate = false;
                if (x < 0. && modes[num].symmetry != SYMMETRY_NONE) {
                    x = -x; if (modes[num].symmetry == SYMMETRY_NEGATIVE) negate = true;
                }
                size_t ix = mesh->tran().findIndex(x);
                if (ix >= xend) ix = xend-1;
                if (ix < xbegin) ix = xbegin;
                if (ix != 0) x -= mesh->tran()[ix-1];
                else if (modes[num].symmetry == SYMMETRY_NONE) x -= mesh->tran()[0];
                dcomplex phasx = exp(- I * kx[ix] * x);
                dcomplex val = modes[num].xfields[ix].F * phasx + modes[num].xfields[ix].B / phasx;
                if (negate) val = - val;
                valx[idx] = val;
            }

            #pragma omp for
            for (size_t idy = 0; idy < rect_mesh.vert().size(); ++idy) {
                double y = rect_mesh.vert()[idy];
                size_t iy = mesh->vert().findIndex(y);
                if (iy >= yend) iy = yend-1;
                if (iy < ybegin) iy = ybegin;
                y -= mesh->vert()[max(int(iy)-1, 0)];
                dcomplex phasy = exp(- I * ky[iy] * y);
                valy[idy] = yfields[iy].F * phasy + yfields[iy].B / phasy;
            }

            if (rect_mesh.getIterationOrder() == RectangularMesh<2>::ORDER_10) {
                #pragma omp for
                for (size_t i1 = 0; i1 < rect_mesh.axis1->size(); ++i1) {
                    double* data = results.data() + i1 * rect_mesh.axis0->size();
                    for (size_t i0 = 0; i0 < rect_mesh.axis0->size(); ++i0) {
                        dcomplex f = valx[i0] * valy[i1];
                        data[i0] = abs2(f);
                    }
                }
            } else {
                #pragma omp for
                for (size_t i0 = 0; i0 < rect_mesh.axis0->size(); ++i0) {
                    double* data = results.data() + i0 * rect_mesh.axis1->size();
                    for (size_t i1 = 0; i1 < rect_mesh.axis1->size(); ++i1) {
                        dcomplex f = valx[i0] * valy[i1];
                        data[i1] = abs2(f);
                    }
                }
            }
        }

        return true;
    }

    return false;
}


DataVector<const Tensor3<dcomplex>> EffectiveIndex2DSolver::getRefractiveIndex(const MeshD<2>& dst_mesh, double lam, InterpolationMethod) {
    this->writelog(LOG_DETAIL, "Getting refractive indices");
    dcomplex ok0 = k0;
    if (!isnan(lam) && lam != 0.) k0 = 2e3*M_PI / lam;
    try { updateCache(); }
    catch(...) { k0 = ok0; throw; }
    k0 = ok0;
    auto target_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    DataVector<Tensor3<dcomplex>> result(dst_mesh.size());
    for (size_t i = 0; i != dst_mesh.size(); ++i) {
        auto point = target_mesh[i];
        size_t x = std::lower_bound(this->mesh->axis0->begin(), this->mesh->axis0->end(), point[0]) - this->mesh->axis0->begin();
        size_t y = std::lower_bound(this->mesh->axis1->begin(), this->mesh->axis1->end(), point[1]) - this->mesh->axis1->begin();
        if (x < xbegin) x = xbegin;
        result[i] = Tensor3<dcomplex>(nrCache[x][y]);
    }
    return result;
}


plask::DataVector<const double> EffectiveIndex2DSolver::getHeat(const MeshD<2>& dst_mesh, plask::InterpolationMethod method) {
}


}}} // namespace plask::solvers::effective
