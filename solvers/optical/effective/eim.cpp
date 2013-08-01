#include <exception>
#include "eim.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

EffectiveIndex2DSolver::EffectiveIndex2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    log_value(dataLog<dcomplex, dcomplex>("Neff", "Neff", "det")),
    have_fields(false),
    recompute_neffs(true),
    stripex(0.),
    polarization(TE),
    symmetry(NO_SYMMETRY),
    vneff(0.),
    outdist(0.1),
    outIntensity(this, &EffectiveIndex2DSolver::getLightIntenisty) {
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
    stripe_root.maxstep = 0.1;
    stripe_root.maxiter = 500;
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
            auto sym = reader.getAttribute("symmetry");
            if (sym) {
                if (*sym == "0" || *sym == "none" ) {
                    symmetry = NO_SYMMETRY;
                }
                else if (*sym == "positive" || *sym == "pos" || *sym == "symmeric" || *sym == "+" || *sym == "+1") {
                    symmetry = SYMMETRY_POSITIVE;;
                }
                else if (*sym == "negative" || *sym == "neg" || *sym == "anti-symmeric" || *sym == "antisymmeric" || *sym == "-" || *sym == "-1") {
                    symmetry = SYMMETRY_NEGATIVE;
                } else throw BadInput(getId(), "Wrong symmetry specification '%1%' in XML", *sym);
            }
            auto wavelength = reader.getAttribute<double>("wavelength");
            if (wavelength) inWavelength.setValue(*wavelength);
            stripex = reader.getAttribute<double>("stripex", stripex);
            vneff = reader.getAttribute<dcomplex>("vneff", vneff);
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
        } else if (param == "mirrors") {
            double R1 = reader.requireAttribute<double>("R1");
            double R2 = reader.requireAttribute<double>("R2");
            mirrors.reset(std::make_pair(R1,R2));
            reader.requireTagEnd();
        } else if (param == "outer") {
            outdist = reader.requireAttribute<double>("distance");
            reader.requireTagEnd();
        } else
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <mode>, <root>, <stripe-root>, or <outer>");
    }
}


std::vector<dcomplex> EffectiveIndex2DSolver::findVeffs(dcomplex neff1, dcomplex neff2, size_t resteps, size_t imsteps, dcomplex eps)
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

    auto results = findZeros(this, [&](const dcomplex& z){return this->detS1(z,nrCache[stripe]);}, neff1, neff2, resteps, imsteps, eps);

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


dcomplex EffectiveIndex2DSolver::computeMode(dcomplex neff)
{
    writelog(LOG_INFO, "Searching for the mode starting from Neff = %1%", str(neff));
    stageOne();
    dcomplex result = RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, root)(neff);
    outNeff = result;
    outNeff.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return result;
}


std::vector<dcomplex> EffectiveIndex2DSolver::findModes(dcomplex neff1, dcomplex neff2, size_t resteps, size_t imsteps, dcomplex eps)
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

    auto results = findZeros(this, [this](dcomplex z){return this->detS(z);}, neff1, neff2, resteps, imsteps, eps);

    if (results.size() != 0) {
        Data2DLog<dcomplex,dcomplex> logger(getId(), "Neffs", "Neff", "det");
        std::string msg = "Found modes at: ";
        for (auto z: results) {
            msg += str(z) + ", ";
            logger(z, detS(z));
        }
        writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
    } else
        writelog(LOG_RESULT, "Did not find any modes");

    return results;
}


void EffectiveIndex2DSolver::setMode(dcomplex neff)
{
    if (!initialized) {
        writelog(LOG_WARNING, "Solver invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    double det = abs(detS(neff));
    if (det > root.tolf_max)
        writelog(LOG_WARNING, "Provided effective index does not correspond to any mode (det = %1%)", det);
    writelog(LOG_INFO, "Setting current mode to %1%", str(neff));
    outNeff = neff;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}


void EffectiveIndex2DSolver::onInitialize()
{
    if (!geometry) throw NoGeometryException(getId());

    // Set default mesh
    if (!mesh) setSimpleMesh();

    xbegin = 0;
    ybegin = 0;
    xend = mesh->axis0.size() + 1;
    yend = mesh->axis1.size() + 1;

    if (geometry->isExtended(Geometry::DIRECTION_TRAN, false) &&
        abs(mesh->axis0[0] - geometry->getChild()->getBoundingBox().lower.c0) < SMALL)
        xbegin = 1;
    if (geometry->isExtended(Geometry::DIRECTION_VERT, false) &&
        abs(mesh->axis1[0] - geometry->getChild()->getBoundingBox().lower.c1) < SMALL)
        ybegin = 1;
    if (geometry->isExtended(Geometry::DIRECTION_TRAN, true) &&
        abs(mesh->axis0[mesh->axis0.size()-1] - geometry->getChild()->getBoundingBox().upper.c0) < SMALL)
        --xend;
    if (geometry->isExtended(Geometry::DIRECTION_VERT, true) &&
        abs(mesh->axis1[mesh->axis1.size()-1] - geometry->getChild()->getBoundingBox().upper.c1) < SMALL)
        --yend;

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(xend, std::vector<dcomplex,aligned_allocator<dcomplex>>(yend));
    epsilons.resize(xend);

    xfields.resize(xend);
    yfields.resize(yend);
}


void EffectiveIndex2DSolver::onInvalidate()
{
    outNeff.invalidate();
    have_fields = false;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/


void EffectiveIndex2DSolver::updateCache()
{
    bool fresh = !initCalculation();

    // Some additional checks
    if (symmetry == SYMMETRY_POSITIVE || symmetry == SYMMETRY_NEGATIVE) {
        if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
            if (fresh) // Make sure we have only positive points
                for (auto x: mesh->axis0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
            if (mesh->axis0[0] == 0.) xbegin = 1;
        } else {
            writelog(LOG_WARNING, "Symmetry reset to NO_SYMMETRY for non-symmetric geometry.");
            symmetry = NO_SYMMETRY;
        }
    }

    if (fresh || inTemperature.changed() || inWavelength.changed() || inGain.changed()) {
        // we need to update something

        k0 = 2e3*M_PI / inWavelength();
        double w = inWavelength();

        RectilinearMesh2D midmesh = *mesh->getMidpointsMesh();
        if (xbegin == 0) {
            if (symmetry != NO_SYMMETRY) midmesh.axis0.addPoint(0.5 * mesh->axis0[0]);
            else midmesh.axis0.addPoint(mesh->axis0[0] - outdist);
        }
        if (xend == mesh->axis0.size()+1)
            midmesh.axis0.addPoint(mesh->axis0[mesh->axis0.size()-1] + outdist);
        if (ybegin == 0)
            midmesh.axis1.addPoint(mesh->axis1[0] - outdist);
        if (yend == mesh->axis1.size()+1)
            midmesh.axis1.addPoint(mesh->axis1[mesh->axis1.size()-1] + outdist);

        writelog(LOG_DEBUG, "Updating refractive indices cache");
        auto temp = inTemperature(midmesh);
        bool need_gain = true;
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
                    if (need_gain) {
                        gain = inGain(midmesh, w);
                        need_gain = false;
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

        outNeff.invalidate();

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
        RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[stripe]);}, log_stripe, stripe_root);
        if (vneff == 0.) {
            dcomplex maxn = *std::max_element(nrCache[stripe].begin(), nrCache[stripe].end(),
                                              [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
            vneff = 0.999 * real(maxn);
        }
        vneff = rootdigger(vneff);

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
        phas = exp(I * ky[ybegin] * (mesh->axis1[ybegin]-mesh->axis1[ybegin-1]));

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
        if (i != mesh->axis1.size())
            phas = exp(I * ky[i] * (mesh->axis1[i]-mesh->axis1[i-1]));

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
        double d = mesh->axis1[i]-mesh->axis1[i-1];
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

void EffectiveIndex2DSolver::normalizeFields(const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx) {

    size_t start;

    double sum = abs2(xfields[xend-1].F) * 0.5 / abs(imag(kx[xend-1]));
    if (symmetry == NO_SYMMETRY) {
        sum += abs2(xfields[xbegin].B) * 0.5 / abs(imag(kx[xbegin]));
        start = xbegin+1;
    } else {
        start = xbegin;
    }

    for (size_t i = start; i < xend-1; ++i) {
        double d = mesh->axis1[i] - ((i == 0)? 0. : mesh->axis1[i-1]);
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
            sum -= imag(xfields[i].F * conj(xfields[i].F) * w_ff +
                        xfields[i].F * conj(xfields[i].B) * w_fb +
                        xfields[i].B * conj(xfields[i].F) * w_bf +
                        xfields[i].B * conj(xfields[i].B) * w_bb
                       );
        }
    }

//     // Consider loss on the mirror
//     double R1, R2;
//     if (mirrors) {
//         std::tie(R1,R2) = *mirrors;
//     } else {
//         const double n = real(vneff);
//         const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
//                         n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
//         R1 = abs((n-n1) / (n+n1));
//         R2 = abs((n-n2) / (n+n2));
//     }


    register dcomplex f = sqrt(1e9 / phys::mu0 / phys::c / sum);  // 1e9 because power in mW and integral computed in µm
    for (size_t i = xbegin; i < xend; ++i) {
        xfields[i] *= f;
    }
}

dcomplex EffectiveIndex2DSolver::detS(const dcomplex& x, bool save)
{
    // Adjust for mirror losses
    dcomplex neff2 = dcomplex(real(x), imag(x)-getMirrorLosses()); neff2 *= neff2;

    std::vector<dcomplex,aligned_allocator<dcomplex>> kx(xend);
    for (size_t i = xbegin; i < xend; ++i) {
        kx[i] = k0 * sqrt(epsilons[i] - neff2);
        if (imag(kx[i]) > 0.) kx[i] = -kx[i];
    }

    Matrix* matrices;
    if (save) matrices = aligned_malloc<Matrix>(xend-1);

    Matrix T = Matrix::eye();
    for (size_t i = xbegin; i < xend-1; ++i) {
        double d;
        if (i != xbegin) d = mesh->axis0[i] - mesh->axis0[i-1];
        else if (symmetry != NO_SYMMETRY) d = mesh->axis0[i];     // we have symmetry, so beginning of the transfer matrix is at the axis
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
        xfields[xend-1] = Field(1., 0.);
        for (size_t i = xend-1; i != xbegin; --i) {
            xfields[i-1] = matrices[i-1].solve(xfields[i]);
        }
        normalizeFields(kx);
        have_fields = true;
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i)
                nrs << "), (" << str(xfields[i].F) << ":" << str(xfields[i].B);
            writelog(LOG_DEBUG, "horizontal fields = [%1%) ]", nrs.str().substr(2));
        }
#endif
        aligned_free(matrices);
    }

    if (symmetry == SYMMETRY_POSITIVE) return T.bf + T.bb;      // B0 = F0   Bn = 0
    else if (symmetry == SYMMETRY_NEGATIVE) return T.bf - T.bb; // B0 = -F0  Bn = 0
    else return T.bb;                                           // F0 = 0    Bn = 0
}


plask::DataVector<const double> EffectiveIndex2DSolver::getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    dcomplex neff = outNeff();

    if (!have_fields) detS(neff, true);

    writelog(LOG_INFO, "Computing field distribution for Neff = %1%", str(neff));

    std::vector<dcomplex,aligned_allocator<dcomplex>> kx(xend);
    dcomplex neff2 = dcomplex(real(neff), imag(neff)-getMirrorLosses()); neff2 *= neff2;
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

    if (!getLightIntenisty_Efficient<RectilinearMesh2D>(dst_mesh, results, kx, ky) &&
        !getLightIntenisty_Efficient<RegularMesh2D>(dst_mesh, results, kx, ky)) {

        #pragma omp parallel for
        for (size_t idx = 0; idx < dst_mesh.size(); ++idx) {
            auto point = dst_mesh[idx];
            double x = point.tran();
            double y = point.vert();

            bool negate = false;
            if (x < 0. && symmetry != NO_SYMMETRY) {
                x = -x; if (symmetry == SYMMETRY_NEGATIVE) negate = true;
            }
            size_t ix = mesh->tran().findIndex(x);
            if (ix >= xend) ix = xend-1;
            if (ix < xbegin) ix = xbegin;
            if (ix != 0) x -= mesh->tran()[ix-1];
            else if (symmetry == NO_SYMMETRY) x -= mesh->tran()[0];
            dcomplex phasx = exp(- I * kx[ix] * x);
            dcomplex val = xfields[ix].F * phasx + xfields[ix].B / phasx;
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

template <typename MeshT>
bool EffectiveIndex2DSolver::getLightIntenisty_Efficient(const plask::MeshD<2>& dst_mesh, DataVector<double>& results,
                                                         const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx,
                                                         const std::vector<dcomplex,aligned_allocator<dcomplex>>& ky)
{
    if (dynamic_cast<const MeshT*>(&dst_mesh)) {

        const MeshT& rect_mesh = dynamic_cast<const MeshT&>(dst_mesh);

        std::vector<dcomplex,aligned_allocator<dcomplex>> valx(rect_mesh.tran().size());
        std::vector<dcomplex,aligned_allocator<dcomplex>> valy(rect_mesh.vert().size());

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (size_t idx = 0; idx < rect_mesh.tran().size(); ++idx) {
                double x = rect_mesh.tran()[idx];
                bool negate = false;
                if (x < 0. && symmetry != NO_SYMMETRY) {
                    x = -x; if (symmetry == SYMMETRY_NEGATIVE) negate = true;
                }
                size_t ix = mesh->tran().findIndex(x);
                if (ix >= xend) ix = xend-1;
                if (ix < xbegin) ix = xbegin;
                if (ix != 0) x -= mesh->tran()[ix-1];
                else if (symmetry == NO_SYMMETRY) x -= mesh->tran()[0];
                dcomplex phasx = exp(- I * kx[ix] * x);
                dcomplex val = xfields[ix].F * phasx + xfields[ix].B / phasx;
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

            if (rect_mesh.getIterationOrder() == MeshT::NORMAL_ORDER) {
                #pragma omp for
                for (size_t i1 = 0; i1 < rect_mesh.axis1.size(); ++i1) {
                    double* data = results.data() + i1 * rect_mesh.axis0.size();
                    for (size_t i0 = 0; i0 < rect_mesh.axis0.size(); ++i0) {
                        dcomplex f = valx[i0] * valy[i1];
                        data[i0] = abs2(f);
                    }
                }
            } else {
                #pragma omp for
                for (size_t i0 = 0; i0 < rect_mesh.axis0.size(); ++i0) {
                    double* data = results.data() + i0 * rect_mesh.axis1.size();
                    for (size_t i1 = 0; i1 < rect_mesh.axis1.size(); ++i1) {
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


}}} // namespace plask::solvers::effective
