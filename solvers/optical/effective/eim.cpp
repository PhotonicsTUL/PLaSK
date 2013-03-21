#include <exception>
#include "eim.h"

using plask::dcomplex;

namespace plask { namespace solvers { namespace effective {

EffectiveIndex2DSolver::EffectiveIndex2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    log_value(dataLog<dcomplex, dcomplex>("Neff", "Neff", "det")),
    have_fields(false),
    old_polarization(TE),
    polarization(TE),
    symmetry(NO_SYMMETRY),
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
    stripe_root.maxstep = 0.5;
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

    // Assign space for refractive indices cache and stripe effective indices
    nrCache.assign(mesh->tran().size()+1, std::vector<dcomplex>(mesh->vert().size()+1));
    stripeNeffs.resize(mesh->tran().size()+1);
}


void EffectiveIndex2DSolver::onInvalidate()
{
    outNeff.invalidate();
    have_fields = false;
    outNeff.fireChanged();
    outIntensity.fireChanged();
}

/********* Here are the computations *********/

/* It would probably be better to use S-matrix method, but for simplicity we use T-matrix */

using namespace Eigen;



void EffectiveIndex2DSolver::stageOne()
{
    bool fresh = !initCalculation();
    bool recompute_neffs = false;

    xbegin = 0;
    ybegin = 0;
    xend = mesh->axis0.size() + 1;
    yend = mesh->axis1.size() + 1;

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

    if (geometry->isExtended(Geometry2DCartesian::DIRECTION_TRAN, false)) xbegin = 1;
    if (geometry->isExtended(Geometry2DCartesian::DIRECTION_VERT, false)) ybegin = 1;
    if (geometry->isExtended(Geometry2DCartesian::DIRECTION_TRAN, true))  --xend;
    if (geometry->isExtended(Geometry2DCartesian::DIRECTION_VERT, true))  --yend;

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
            size_t tx = ix - xbegin;
            for (size_t iy = ybegin; iy < yend; ++iy) {
                size_t ty = iy - ybegin;
                double T = temp[midmesh.index(tx,ty)];
                auto point = midmesh(tx, ty);
                auto roles = geometry->getRolesAt(point);
                if (roles.find("QW") == roles.end() && roles.find("QD") == roles.end() && roles.find("gain") == roles.end())
                    nrCache[ix][iy] = geometry->getMaterial(point)->Nr(w, T);
                else {  // we ignore the material absorption as it should be considered in the gain already
                    double g = gain[midmesh.index(tx, ty)];
                    nrCache[ix][iy] = dcomplex( real(geometry->getMaterial(point)->Nr(w, T)),
                                                w * g * 7.95774715459e-09 );
                }
            }
        }
        recompute_neffs = true;
    }

    if (recompute_neffs || polarization != old_polarization) {

        old_polarization = polarization;

        // Compute effective indices for all stripes
        std::exception_ptr error; // needed to handle exceptions from OMP loop
        #pragma omp parallel for
        for (size_t i = xbegin; i < xend; ++i) {
            if (error != std::exception_ptr()) continue; // just skip loops after error
            try {
                writelog(LOG_DETAIL, "Computing effective index for vertical stripe %1% (polarization %2%)", i-xbegin, (polarization==TE)?"TE":"TM");
#               ifndef NDEBUG
                    std::stringstream nrs; for (size_t j = ybegin; j != yend; ++j) nrs << ", " << str(nrCache[i][j]);
                    writelog(LOG_DEBUG, "Nr[%1%] = [%2% ]", i-xbegin, nrs.str().substr(1));
#               endif

                dcomplex same_val = nrCache[i][ybegin];
                bool all_the_same = true;
                for (size_t j = ybegin; j != yend; ++j) if (nrCache[i][j] != same_val) { all_the_same = false; break; }
                if (all_the_same) {
                    stripeNeffs[i] = same_val;
                } else {
                    Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[%1%]", i-xbegin), "neff", "det");
                    RootDigger rootdigger(*this, [&](const dcomplex& x){return this->detS1(x,nrCache[i]);}, log_stripe, stripe_root);
                    dcomplex maxn = *std::max_element(nrCache[i].begin(), nrCache[i].end(),
                                                      [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
                    stripeNeffs[i] = rootdigger.getSolution(0.999 * real(maxn));
                }
            } catch (...) {
                #pragma omp critical
                error = std::current_exception();
            }
        }
        if (error != std::exception_ptr()) std::rethrow_exception(error);

#       ifndef NDEBUG
            std::stringstream nrs; for (size_t i = xbegin; i < stripeNeffs.size(); ++i) nrs << ", " << str(stripeNeffs[i]);
            writelog(LOG_DEBUG, "stripes neffs = [%1% ]", nrs.str().substr(1));
#       endif
    }
}


dcomplex EffectiveIndex2DSolver::detS1(const plask::dcomplex& x, const std::vector<dcomplex>& NR)
{
    std::vector<dcomplex> beta(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        beta[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
        if (imag(beta[i]) > 0.) beta[i] = -beta[i];
    }

    dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S

    dcomplex phas = 1.;
    if (ybegin != 0)
        phas = exp(I * beta[ybegin] * (mesh->axis1[ybegin]-mesh->axis1[ybegin-1]));

    for (size_t i = ybegin; i < yend-1; ++i) {
        // Compute shift inside one layer
        s1 *= phas;
        s3 *= phas * phas;
        s4 *= phas;
        // Compute matrix after boundary
        dcomplex f = (polarization==TM)? NR[i]/NR[i+1] : 1.;
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
        if (i != mesh->axis1.size())
            phas = exp(I * beta[i+1] * (mesh->axis1[i+1]-mesh->axis1[i]));
    }

    return s4 - s2*s3/s1;
}


dcomplex EffectiveIndex2DSolver::detS(const dcomplex& x)
{
    // Adjust for mirror losses
    dcomplex neff = dcomplex(real(x), imag(x)-getMirrorLosses(x));

    std::vector<dcomplex> beta(xend);
    for (size_t i = xbegin; i < xend; ++i) {
        beta[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
        if (imag(beta[i]) > 0.) beta[i] = -beta[i];
    }

    dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S

    dcomplex phas = 1.;

    if (symmetry != NO_SYMMETRY) // we have symmetry, so begin of the transfer matrix is at the axis
        phas = exp(I * beta[xbegin] * mesh->axis0[xbegin]);
    else if (xbegin != 0)
        phas = exp(I * beta[xbegin] * (mesh->axis0[xbegin]-mesh->axis0[xbegin-1]));

    for (size_t i = xbegin; i < xend-1; ++i) {
        // Compute shift inside one layer
        s1 *= phas;
        s3 *= phas * phas;
        s4 *= phas;
        // Compute matrix after boundary
        dcomplex f = (polarization==TE)? stripeNeffs[i]/stripeNeffs[i+1] : 1.;
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
    }

    // Rn = | T00 T01 | R0
    // Ln = | T10 T11 | L0
    if (symmetry == SYMMETRY_POSITIVE) return s4 - (s2*s3 - s3) / s1;
    else if (symmetry == SYMMETRY_NEGATIVE) return s4 - (s2*s3 + s3) / s1;
    else return s4 - s2*s3 / s1;
}



plask::DataVector<const double> EffectiveIndex2DSolver::getLightIntenisty(const plask::MeshD<2>& dst_mesh, plask::InterpolationMethod)
{
    this->writelog(LOG_DETAIL, "Getting light intensity");

    if (!outNeff.hasValue()) throw NoValue(OpticalIntensity::NAME);

    dcomplex neff = outNeff();

    writelog(LOG_INFO, "Computing field distribution for Neff = %1%", str(neff));

    std::vector<dcomplex> betax(xend);
    for (size_t i = 0; i < xend; ++i) {
        betax[i] = k0 * sqrt(stripeNeffs[i]*stripeNeffs[i] - neff*neff);
        if (imag(betax[i]) > 0.) betax[i] = -betax[i];
    }
    if (!have_fields) {
        auto fresnelX = [&](size_t i) -> Matrix2cd {
            dcomplex f =  (polarization==TE)? stripeNeffs[i]/stripeNeffs[i+1] :  1.;
            dcomplex n = 0.5 * betax[i+1]/betax[i] * f*f;
            Matrix2cd M; M << (0.5+n), (0.5-n),
                              (0.5-n), (0.5+n);
            return M;
        };
        fieldX.resize(xend);
        fieldX[xend-1] << 1., 0;
        fieldWeights.resize(xend);
        fieldWeights[xend-1] = 0.;
        for (ptrdiff_t i = xend-2; i >= 0; --i) {
            fieldX[i].noalias() = fresnelX(i) * fieldX[i+1];
            double d = (symmetry == NO_SYMMETRY)? mesh->tran()[i] - mesh->tran()[max(int(i)-1, 0)] :
                       (i == 0)? mesh->tran()[0] : mesh->tran()[i] - mesh->tran()[i-1];
            dcomplex b = betax[i];
            dcomplex phas = exp(- I * b * d);
            DiagonalMatrix<dcomplex, 2> P;
            P.diagonal() << 1./phas, phas;  // we propagate backward
            fieldX[i] = P * fieldX[i];
            // Compute density of the field is stored in the i-th layer
            dcomplex w_ff, w_bb, w_fb, w_bf;
            if (d != 0.) {
                if (imag(b) != 0) { dcomplex bb = b - conj(b);
                    w_ff = (exp(-I*d*bb) - 1.) / bb;
                    w_bb = (exp(+I*d*bb) - 1.) / bb;
                } else w_ff = w_bb = dcomplex(0., -d);
                if (real(b) != 0) { dcomplex bb = b + conj(b);
                    w_fb = (exp(-I*d*bb) - 1.) / bb;
                    w_bf = (exp(+I*d*bb) - 1.) / bb;
                } else w_ff = w_bb = dcomplex(0., -d);
                fieldWeights[i] = -imag(  fieldX[i][0] * conj(fieldX[i][0]) * w_ff
                                        - fieldX[i][1] * conj(fieldX[i][1]) * w_bb
                                        + fieldX[i][0] * conj(fieldX[i][1]) * w_fb
                                        - fieldX[i][1] * conj(fieldX[i][0]) * w_bb);
            } else {
                fieldWeights[i] = 0.;
            }
        }
        double sumw = 0; for (const double& w: fieldWeights) sumw += w;
        double factor = 1./sumw; for (double& w: fieldWeights) w *= factor;
#       ifndef NDEBUG
            std::stringstream weightss; for (size_t i = xbegin; i < xend; ++i) weightss << ", " << str(fieldWeights[i]);
            writelog(LOG_DEBUG, "field confinement in stripes = [%1% ]", weightss.str().substr(1));
#       endif
    }

    size_t mid_x = std::max_element(fieldWeights.begin(), fieldWeights.end()) - fieldWeights.begin();
    // double max_val = 0.;
    // for (size_t i = 1; i != xend; ++i) { // Find stripe with maximum weight that has non-constant refractive indices
    //     if (fieldWeights[i] > max_val) {
    //         dcomplex same_val = nrCache[i].front(); bool all_the_same = true;
    //         for (auto n: nrCache[i]) if (n != same_val) { all_the_same = false; break; }
    //         if (!all_the_same) {
    //             max_val = fieldWeights[i];
    //             mid_x = i;
    //         }
    //     }
    // }
    writelog(LOG_DETAIL, "Vertical field distribution taken from stripe %1%", mid_x-xbegin);
    std::vector<dcomplex> betay(yend);
    bool all_the_same = true; dcomplex same_n = nrCache[mid_x][0];
    for (size_t j = ybegin; j != yend; ++j) if (nrCache[mid_x][j] != same_n) { all_the_same = false; break; }
    if (all_the_same) {
        betay.assign(yend, 0.);
    } else {
        for (size_t i = ybegin; i < yend; ++i) {
            betay[i] = k0 * sqrt(nrCache[mid_x][i]*nrCache[mid_x][i] - stripeNeffs[mid_x]*stripeNeffs[mid_x]);
            if (imag(betay[i]) > 0.) betay[i] = -betay[i];
        }
    }
    if (!have_fields) {
        if (all_the_same) {
            fieldY.assign(yend, 0.5 * Vector2cd::Ones(2));
        } else {
            auto fresnelY = [&](size_t i) -> Matrix2cd {
                dcomplex f =  (polarization==TM)? nrCache[mid_x][i]/nrCache[mid_x][i+1] :  1.;
                dcomplex n = 0.5 * betay[i+1]/betay[i] * f*f;
                Matrix2cd M; M << (0.5+n), (0.5-n),
                                  (0.5-n), (0.5+n);
                return M;
            };
            fieldY.resize(yend);
            fieldY[yend-1] << 1., 0;
            for (ptrdiff_t i = yend-2; i >= 0; --i) {
                fieldY[i].noalias() = fresnelY(i) * fieldY[i+1];
                double d = mesh->vert()[i] - mesh->vert()[max(int(i)-1, 0)];
                dcomplex phas = exp(- I * betay[i] * d);
                DiagonalMatrix<dcomplex, 2> P;
                P.diagonal() << 1./phas, phas;  // we propagate backward
                fieldY[i] = P * fieldY[i];
            }
        }
    }

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
            dcomplex val = fieldX[ix][0] * phasx + fieldX[ix][1] / phasx;
            if (negate) val = - val;

            size_t iy = mesh->vert().findIndex(y);
            if (iy >= yend) iy = yend-1;
            if (iy < ybegin) iy = ybegin;
            y -= mesh->vert()[max(int(iy)-1, 0)];
            dcomplex phasy = exp(- I * betay[iy] * y);
            val *= fieldY[iy][0] * phasy + fieldY[iy][1] / phasy;

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
                dcomplex val = fieldX[ix][0] * phasx + fieldX[ix][1] / phasx;
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
                valy[idy] = fieldY[iy][0] * phasy + fieldY[iy][1] / phasy;
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
