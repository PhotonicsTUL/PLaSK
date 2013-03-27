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
    root.tolx = 1.0e-6;
    root.tolf_min = 1.0e-8;
    root.tolf_max = 1.0e-5;
    root.maxstep = 0.1;
    root.maxiter = 500;
    stripe_root.tolx = 1.0e-6;
    stripe_root.tolf_min = 1.0e-8;
    stripe_root.tolf_max = 1.0e-5;
    stripe_root.maxstep = 0.05;
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

dcomplex EffectiveIndex2DSolver::computeMode(dcomplex neff)
{
    writelog(LOG_INFO, "Searching for the mode starting from Neff = %1%", str(neff));
    stageOne();
    dcomplex result = RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, root).getSolution(neff);
    outNeff = result;
    outNeff.fireChanged();
    outIntensity.fireChanged();
    have_fields = false;
    return result;
}



std::vector<dcomplex> EffectiveIndex2DSolver::findModes(dcomplex neff1, dcomplex neff2, unsigned steps, unsigned nummodes)
{
    writelog(LOG_INFO, "Searching for the modes for Neff between %1% and %2%", str(neff1), str(neff2));
    stageOne();
    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, root)
            .searchSolutions(neff1, neff2, steps, 0, nummodes);
}

std::vector<dcomplex> EffectiveIndex2DSolver::findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps)
{
    writelog(LOG_INFO, "Searching for the approximate modes for Neff between %1% and %2%", str(neff1), str(neff2));
    stageOne();

    return RootDigger(*this, [this](const dcomplex& x){return this->detS(x);}, log_value, root)
            .findMap(neff1, neff2, steps, 0);
}


void EffectiveIndex2DSolver::setMode(dcomplex neff)
{
    if (!initialized) {
        writelog(LOG_WARNING, "Solver invalidated or not initialized, so performing some initial computations");
        stageOne();
    }
    double det = abs(detS(neff));
    if (det > root.tolf_max) throw BadInput(getId(), "Provided effective index does not correspond to any mode (det = %1%)", det);
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
    nrCache.assign(xend, std::vector<dcomplex>(yend));
    neffs.resize(xend);
}


void EffectiveIndex2DSolver::onInvalidate()
{
    outNeff.invalidate();
    have_fields = false;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/


void EffectiveIndex2DSolver::stageOne()
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

    if (fresh || inTemperature.changed || inWavelength.changed || inGain.changed) {
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
        auto gain = inGain(midmesh, w);

        for (size_t ix = xbegin; ix < xend; ++ix) {
            for (size_t iy = ybegin; iy < yend; ++iy) {
                size_t idx = midmesh.index(ix-xbegin, iy-ybegin);
                double T = temp[idx];
                auto point = midmesh[idx];
                auto roles = geometry->getRolesAt(point);
                if (roles.find("QW") == roles.end() && roles.find("QD") == roles.end() && roles.find("gain") == roles.end())
                    nrCache[ix][iy] = geometry->getMaterial(point)->Nr(w, T);
                else {  // we ignore the material absorption as it should be considered in the gain already
                    double g = gain[idx];
                    nrCache[ix][iy] = dcomplex( real(geometry->getMaterial(point)->Nr(w, T)),
                                                w * g * 7.95774715459e-09 );
                }
            }
        }
        recompute_neffs = true;
    }

    if (recompute_neffs) {

        // Compute effective index of main stripe
        size_t stripe = mesh->tran().findIndex(stripex);
        if (stripe < xbegin) stripe = xbegin;
        else if (stripe >= xend) stripe = xend-1;
        writelog(LOG_DETAIL, "Computing effective index for vertical stripe %1% (polarization %2%)", stripe, (polarization==TE)?"TE":"TM");
#ifndef NDEBUG
        {
            std::stringstream nrs; for (ptrdiff_t j = yend-1; j >= ptrdiff_t(ybegin); --j) nrs << ", " << str(nrCache[stripe][j]);
            writelog(LOG_DEBUG, "Nr[%1%] = [%2% ]", stripe, nrs.str().substr(1));
        }
#endif
        Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[%1%]", stripe-xbegin), "neff", "det");
        RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[stripe]);}, log_stripe, stripe_root);
        if (vneff == 0.) {
            dcomplex maxn = *std::max_element(nrCache[stripe].begin(), nrCache[stripe].end(),
                                              [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
            vneff = 0.999 * real(maxn);
        }
        vneff = rootdigger.getSolution(vneff);

        // Compute field weights
        computeWeights(stripe);

        // Compute effective indices
        for (size_t i = xbegin; i < xend; ++i) {
            if (i == stripex) {
                neffs[i] = vneff;
            } else {
                neffs[i] = 0;
                for (size_t j = ybegin; j < yend; ++j) {
                    neffs[i] += weights[j] * nrCache[i][j];
                }
            }
        }
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i) nrs << ", " << str(neffs[i]);
            writelog(LOG_DEBUG, "horizontal neffs = [%1% ]", nrs.str().substr(1));
        }
#endif
        recompute_neffs = false;
    }
}

void EffectiveIndex2DSolver::computeWeights(size_t stripe)
{
    // Compute fields
    detS1(vneff, nrCache[stripe], true);

    double sum = 0.;
    weights.resize(yend);
    weights[ybegin] = 0.;
    weights[yend-1] = 0.;

    for (size_t i = ybegin+1; i < yend-1; ++i) {
        double d = mesh->axis1[i]-mesh->axis1[i];
        dcomplex b = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - vneff*vneff); if (imag(b) > 0.) b = -b;
        dcomplex w_ff, w_bb, w_fb, w_bf;
        if (d != 0.) {
            if (abs(imag(b)) > SMALL) {
                dcomplex bb = b - conj(b);
                w_ff = (exp(-I*d*bb) - 1.) / bb;
                w_bb = (exp(+I*d*bb) - 1.) / bb;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            if (abs(real(b)) > SMALL) {
                dcomplex bb = b + conj(b);
                w_fb = (exp(-I*d*bb) - 1.) / bb;
                w_bf = (exp(+I*d*bb) - 1.) / bb;
            } else
                w_ff = w_bb = dcomplex(0., -d);
            weights[i] = -imag(fieldY[i].F * conj(fieldY[i].F) * w_ff
                                    - fieldY[i].B * conj(fieldY[i].F) * w_bf
                                    + fieldY[i].F * conj(fieldY[i].B) * w_fb
                                    - fieldY[i].B * conj(fieldY[i].B) * w_bb);
        } else
            weights[i] = 0.;
        sum += weights[i];
    }

    sum = 1. / sum;
    for (size_t i = ybegin; i < yend; ++i) {
        weights[i] *= sum;
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i) nrs << ", " << str(weights[i]);
            writelog(LOG_DEBUG, "vertical weights = [%1%) ]", nrs.str().substr(2));
        }
#endif
    }

}

dcomplex EffectiveIndex2DSolver::detS1(const plask::dcomplex& x, const std::vector<dcomplex>& NR, bool save)
{
    double maxf = 0.;
    if (save) {
        fieldY.resize(yend);
        fieldY[ybegin] = Field(0., 1.);
    }

    std::vector<dcomplex> beta(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
        if (imag(beta[i]) > 0.) beta[i] = -beta[i];
    }

    dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S

    dcomplex phas = 1.;
    if (ybegin != 0)
        phas = exp(I * beta[ybegin] * (mesh->axis1[ybegin]-mesh->axis1[ybegin-1]));

    for (size_t i = ybegin+1; i < yend; ++i) {
        // Compute shift inside one layer
        s1 *= phas;
        s3 *= phas * phas;
        s4 *= phas;
        // Compute matrix after boundary
        dcomplex f = (polarization==TM)? NR[i-1]/NR[i] : 1.;
        dcomplex p = 0.5 + 0.5 * beta[i] / beta[i-1] * f*f;
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
            phas = exp(I * beta[i] * (mesh->axis1[i]-mesh->axis1[i-1]));

        // Compute fields and weights in the next layer
        if (save) {
            dcomplex F = -s2/s1, B = (s1*s4-s2*s3)/s1;    // Assume  F0 = 0  B0 = 1
            maxf = max(maxf, abs(F.real()));
            maxf = max(maxf, abs(B.real()));
            fieldY[i] = Field(F, B);
        }
    }

    if (save) {
        maxf = 1. / maxf;
        for (size_t i = ybegin; i < yend; ++i) fieldY[i] *= maxf;
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i) nrs << "), (" << str(fieldY[i].F) << ":" << str(fieldY[i].B);
            writelog(LOG_DEBUG, "vertical fields = [%1%) ]", nrs.str().substr(2));
        }
#endif
    }

    return s1*s4 - s2*s3;
}


dcomplex EffectiveIndex2DSolver::detS(const dcomplex& x, bool save)
{
    double maxf = 0.;
    if (save) {
        fieldX.resize(xend);
        if (symmetry == SYMMETRY_POSITIVE)      { fieldX[xbegin] = Field( 1., 1.); }    // F0 = B0
        else if (symmetry == SYMMETRY_NEGATIVE) { fieldX[xbegin] = Field(-1., 1.); }    // F0 = -B0
        else                                    { fieldX[xbegin] = Field( 0., 1.); }    // F0 = 0  B0 = 1
    }

    // Adjust for mirror losses
    dcomplex neff = dcomplex(real(x), imag(x)-getMirrorLosses(x));

    std::vector<dcomplex> beta(xend);
    for (size_t i = xbegin; i < xend; ++i) {
        beta[i] = k0 * sqrt(neffs[i]*neffs[i] - neff*neff);
        if (imag(beta[i]) > 0.) beta[i] = -beta[i];
    }

    dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S

    dcomplex phas = 1.;

    if (symmetry != NO_SYMMETRY) // we have symmetry, so begin of the transfer matrix is at the axis
        phas = exp(I * beta[xbegin] * mesh->axis0[xbegin]);
    else if (xbegin != 0)
        phas = exp(I * beta[xbegin] * (mesh->axis0[xbegin]-mesh->axis0[xbegin-1]));

    for (size_t i = xbegin+1; i < xend; ++i) {
        // Compute shift inside one layer
        s1 *= phas;
        s3 *= phas * phas;
        s4 *= phas;
        // Compute matrix after boundary
        dcomplex f = (polarization==TE)? neffs[i]/neffs[i+1] : 1.;
        dcomplex p = 0.5 + 0.5 * beta[i+1] / beta[i] * f*f;
        dcomplex m = 1.0 - p;
        dcomplex chi = 1. / (p - m * s3);
        // F0 = [ (-m*m + p*p)*chi*s1  m*s1*s4*chi + s2 ] [ F2 ]
        // B2 = [ (-m + p*s3)*chi      s4*chi           ] [ B0 ]
        s2 += s1*m*chi*s4;
        s1 *= (p*p - m*m) * chi;
        s3  = (p*s3-m) * chi;
        s4 *= chi;
        // Compute phase shift for the next step
        if (i != mesh->axis0.size())
            phas = exp(I * beta[i+1] * (mesh->axis0[i+1]-mesh->axis0[i]));

        // Compute fields and weights in the next layer
        if (save) {
            // Fn = 1/s1 F0 - s2/s1 B0
            // Bn = s3/s1 F0 + (s1s4-s2s3)/s1 B0
            dcomplex F, B;
            if (symmetry == SYMMETRY_POSITIVE)      { F = ( 1.-s2) / s1; B = s4 - (s2*s3 - s3) / s1; }  // F0 = B0
            else if (symmetry == SYMMETRY_NEGATIVE) { F = (-1.-s2) / s1; B = s4 - (s2*s3 + s3) / s1; }  // F0 = -B0
            else                                    { F = -s2/s1; B = (s1*s4-s2*s3)/s1; }               // F0 = 0  B0 = 1
            maxf = max(maxf, abs(F.real()));
            maxf = max(maxf, abs(B.real()));
            fieldX[i] = Field(F, B);
        }
    }

    // Rn = | T00 T01 | R0
    // Ln = | T10 T11 | L0
    if (symmetry == SYMMETRY_POSITIVE) return s4 - (s2*s3 - s3) / s1;
    else if (symmetry == SYMMETRY_NEGATIVE) return s4 - (s2*s3 + s3) / s1;
    else return s4 - s2*s3 / s1;

    if (save) {
        maxf = 1. / maxf;
        for (size_t i = ybegin; i < yend; ++i) fieldY[i] *= maxf;
#ifndef NDEBUG
        {
            std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i) nrs << "), (" << str(fieldX[i].F) << ":" << str(fieldX[i].B);
            writelog(LOG_DEBUG, "horizontal fields = [%1%) ]", nrs.str().substr(2));
        }
#endif
    }


}


plask::DataVector<const double> EffectiveIndex2DSolver::getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    dcomplex neff = outNeff();

    writelog(LOG_INFO, "Computing field distribution for Neff = %1%", str(neff));

    std::vector<dcomplex> betax(xend);
    for (size_t i = 0; i < xend; ++i) {
        betax[i] = k0 * sqrt(neffs[i]*neffs[i] - neff*neff);
        if (imag(betax[i]) > 0.) betax[i] = -betax[i];
    }

    size_t stripe = mesh->tran().findIndex(stripex);
    if (stripe < xbegin) stripe = xbegin;
    else if (stripe >= xend) stripe = xend-1;


    std::vector<dcomplex> betay(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        betay[i] = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - vneff*vneff);
        if (imag(betay[i]) > 0.) betay[i] = -betay[i];
    }

    if (!have_fields) detS(neff, true);

    DataVector<double> results(dst_mesh.size());

    if (!getLightIntenisty_Efficient<RectilinearMesh2D>(dst_mesh, results, betax, betay) &&
        !getLightIntenisty_Efficient<RegularMesh2D>(dst_mesh, results, betax, betay)) {

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
            dcomplex phasx = exp(- I * betax[ix] * x);
            dcomplex val = fieldX[ix].F * phasx + fieldX[ix].B / phasx;
            if (negate) val = - val;

            size_t iy = mesh->vert().findIndex(y);
            if (iy >= yend) iy = yend-1;
            if (iy < ybegin) iy = ybegin;
            y -= mesh->vert()[max(int(iy)-1, 0)];
            dcomplex phasy = exp(- I * betay[iy] * y);
            val *= fieldY[iy].F * phasy + fieldY[iy].B / phasy;

            results[idx] = real(abs2(val));
        }
    }

    // Normalize results to make maximum value equal to one
    double factor = 1. / *std::max_element(results.begin(), results.end());
    for (double& val: results) val *= factor;

    return results;
}

template <typename MeshT>
bool EffectiveIndex2DSolver::getLightIntenisty_Efficient(const plask::MeshD<2>& dst_mesh, DataVector<double>& results,
                                                         const std::vector<dcomplex>& betax, const std::vector<dcomplex>& betay)
{
    if (dynamic_cast<const MeshT*>(&dst_mesh)) {

        const MeshT& rect_mesh = dynamic_cast<const MeshT&>(dst_mesh);

        std::vector<dcomplex> valx(rect_mesh.tran().size());
        std::vector<dcomplex> valy(rect_mesh.vert().size());

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
                dcomplex phasx = exp(- I * betax[ix] * x);
                dcomplex val = fieldX[ix].F * phasx + fieldX[ix].B / phasx;
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
                dcomplex phasy = exp(- I * betay[iy] * y);
                valy[idy] = fieldY[iy].F * phasy + fieldY[iy].B / phasy;
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
