#include "expansionfdcyl.h"
#include "solverfdcyl.h"

#define SOLVER static_cast<CylindersSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionCylinders::ExpansionCylinders(CylindersSolverCyl* solver): Expansion(solver), m(1),
                                                           initialized(false)
{
}


size_t ExpansionCylinders::matrixSize() const {
    return 2 * (SOLVER->mesh->size() - 1);
}


void ExpansionCylinders::init()
{
    size_t nlayers = solver->lcount;
    diagonals.assign(nlayers, false);
    epsilons.resize(nlayers);

    // Fill-in mu
    raxis = SOLVER->mesh->getMidpointAxis();
    size_t n = raxis->size();
    mu.reset(n, Tensor3<dcomplex>(1.));
    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
        auto bbox = SOLVER->geometry->getChild()->getBoundingBox();
        double shift = max(-bbox.lower.c0, bbox.upper.c0) + SOLVER->pml.dist;
        if (raxis->at(raxis->size()-1) <= shift + SOLVER->pml.size/2.)
            SOLVER->writelog(LOG_WARNING, "Mesh does not span until the end of PMLs ({:.3f}/{:.3f})",
                raxis->at(raxis->size()-1), shift + SOLVER->pml.size);
        for (size_t i = 0; i != n; ++i) {
            double r = abs(raxis->at(i));
            if (r >= shift) {
                dcomplex f = 1. + (SOLVER->pml.factor-1.) * pow((r-shift)/SOLVER->pml.size, SOLVER->pml.order);
                mu[i] = Tensor3<dcomplex>(f, 1./f, f);
            }
        }
    }

    initialized = true;
}


void ExpansionCylinders::reset()
{
    raxis.reset();
    epsilons.clear();
    mu.reset();
    initialized = false;
}



void ExpansionCylinders::prepareIntegrals(double lam, double glam) {
    auto mesh = plask::make_shared<RectangularMesh<2>>(raxis, solver->verts, RectangularMesh<2>::ORDER_01);
    temperature = SOLVER->inTemperature(mesh);
    gain_connected = SOLVER->inGain.hasProvider();
    if (gain_connected) {
        if (isnan(glam)) glam = lam;
        gain = SOLVER->inGain(mesh, glam);
    }
}

void ExpansionCylinders::cleanupIntegrals(double, double) {
    temperature.reset();
    gain.reset();
}

void ExpansionCylinders::layerIntegrals(size_t layer, double lam, double glam) {
    if (isnan(real(k0)) || isnan(imag(k0)))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();

    #ifndef NDEBUG
        #if defined(OPENMP_FOUND)
            SOLVER->writelog(LOG_DETAIL, "Caching epsilons for layer {:d}/{:d} in thread {:d}",
                            layer, solver->lcount, omp_get_thread_num());
        #else
            SOLVER->writelog(LOG_DETAIL, "Caching epsilons for layer {:d}/{:d}",
                            layer, solver->lcount);
        #endif
    #endif

    if (isnan(lam))
        throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    size_t nr = raxis->size();

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    double matz;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            matz = solver->verts->at(i);
            break;
        }
    }

    epsilons[layer].reset(nr);

    // For checking if the layer is uniform
    diagonals[layer] = true;

    // Compute and store permittivity
    for (size_t ri = 0; ri != nr; ++ri) {
        double r = raxis->at(ri);

        Tensor3<dcomplex> eps;
        {
            OmpLockGuard<OmpNestLock> lock; // this must be declared before `material` to guard its destruction
            auto material = geometry->getMaterial(vec(r, matz));
            double T = 0., W = 0.;
            for (size_t k = 0, v = ri * solver->verts->size(); k != solver->verts->size(); ++v, ++k) {
                if (solver->stack[k] == layer) {
                    double w = (k == 0 || k == solver->verts->size()-1)?
                            1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                    T += w * temperature[v]; W += w;
                }
            }
            T /= W;
            lock = material->lock();
            eps = material->NR(lam, T);
            if (isnan(eps.c00) || isnan(eps.c11) || isnan(eps.c22) || isnan(eps.c01))
                throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                               material->name(), lam, T);
        }
        if (eps.c01 != 0.)
            throw BadInput(solver->getId(), "Non-diagonal anisotropy not allowed for this solver");
        if (gain_connected &&  solver->lgained[layer]) {
            auto roles = geometry->getRolesAt(vec(r, matz));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                Tensor2<double> g = 0.; double W = 0.;
                for (size_t k = 0, v = ri * solver->verts->size(); k != solver->verts->size(); ++v, ++k) {
                    if (solver->stack[k] == layer) {
                        double w = (k == 0 || k == solver->verts->size()-1)?
                                1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                        g += w * gain[v]; W += w;
                    }
                }
                Tensor2<double> ni = glam * g/W * (0.25e-7/PI);
                eps.c00.imag(ni.c00); eps.c11.imag(ni.c00); eps.c22.imag(ni.c11);
            }
        }
        eps.sqr_inplace();

        epsilons[layer][ri] = eps;
        if (diagonals[layer] && epsilons[layer][ri] != epsilons[layer][0]) diagonals[layer] = false;
    }

    // Add PMLs
    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
        diagonals[layer] = false;
        auto bbox = SOLVER->geometry->getChild()->getBoundingBox();
        double shift = max(-bbox.lower.c0, bbox.upper.c0) + SOLVER->pml.dist;
        for (size_t i = 0; i != nr; ++i) {
            double r = abs(raxis->at(i));
            if (r >= shift) {
                dcomplex f = 1. + (SOLVER->pml.factor-1.) * pow((r-shift)/SOLVER->pml.size, SOLVER->pml.order);
                epsilons[layer][i].c00 *= f;
                epsilons[layer][i].c11 /= f;
                epsilons[layer][i].c22 *= f;
            }
        }
    }

    if (diagonals[layer])
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
}


void ExpansionCylinders::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    size_t N = raxis->size();
    size_t N1 = N - 1;
    assert(RE.rows() == 2*N);
    assert(RE.cols() == 2*N);
    assert(RH.rows() == 2*N);
    assert(RH.cols() == 2*N);

    std::fill(RE.begin(), RE.end(), dcomplex(0.));
    std::fill(RH.begin(), RH.end(), dcomplex(0.));

    const double m = this->m? this->m : 1.;
    const dcomplex k0m = k0 / m, mk0 = m * k0;

    //The first element
    {
        const double r = 1. / raxis->at(0);
        const double r2 = r * r;
        const double mr = this->m? m * r : 0.;
        const double mr2 = this->m? m * r2 : 0.;

        const size_t ier = iEr(0), iep = iEp(0), ihr = iHr(0), ihp = iHp(0);

        const dcomplex fe = 1. / (k0 * epsilons[layer][0].c22),
                       fh = 1. / (k0 * mu[0].c22);

        double D, Dp, Dm, DD, DDp, DDm;

        if (m == 1) {
            const double dm = 2 * raxis->at(0), dp = raxis->at(1) - raxis->at(0);
            D = (dp-dm) / (dm*dp) - dp/dm / (dm+dp); Dp = dm/dp / (dm+dp);
            DD = - 2. / (dm*dp) + 2. / (dm+dp) / dm; DDp = 2. / (dm+dp) / dp;
        } else {
            const double dm = raxis->at(0), dp = raxis->at(1) - raxis->at(0);
            D = (dp-dm) / (dm*dp); Dp = dm/dp / (dm+dp);
            DD = - 2. / (dm*dp); DDp = 2. / (dm+dp) / dp;
        }

        RE(ier, ihp) += m * fe * (-r2 + r * D + DD) + mk0 * mu[0].c00;
        RE(ier, ihp+i1) += m * fe * (r * Dp + DDp);
        RE(ier, ihr) += fe * mr * (-r + D);
        RE(ier, ihr+i1) += fe * mr * Dp;

        RE(iep, ihp) += fe * mr * (r + D);
        RE(iep, ihp+i1) += fe * mr * Dp;
        RE(iep, ihr) += fe * mr2 - k0m * mu[0].c11;

        RH(ihp, ier) += - fh * mr2 + k0m * epsilons[layer][0].c11;
        RH(ihp, iep) += fh * mr * (r + D);
        RH(ihp, iep+i1) += fh * mr * Dp;

        RH(ihr, ier) += fh * mr * (-r + D);
        RH(ihr, ier+i1) += fh * mr * Dp;
        RH(ihr, iep) += m * fh * (r2 - r * D - DD) - mk0 * epsilons[layer][0].c00;
        RH(ihr, iep+i1) += m * fh * (- r * Dp - DDp);
    }

    for (size_t i = 1; i != N1; ++i) {
        size_t ier = iEr(i), iep = iEp(i), ihr = iHr(i), ihp = iHp(i);

        const double r = 1. / raxis->at(i);
        const double r2 = r * r;
        const double mr = this->m? m * r : 0.;
        const double mr2 = this->m? m * r2 : 0.;

        const dcomplex fe = 1. / (k0 * epsilons[layer][i].c22),
                       fh = 1. / (k0 * mu[i].c22);

        const double dm = raxis->at(i) - raxis->at(i-1), dp = raxis->at(i+1) - raxis->at(i);
        const double D = (dp-dm) / (dm*dp), Dp = dm/dp / (dm+dp), Dm = - dp/dm / (dm+dp);
        const double DD = - 2. / (dm*dp), DDp = 2. / (dm+dp) / dp, DDm = 2. / (dm+dp) / dm;

        RE(ier, ihp) += m * fe * (-r2 + r * D + DD) + mk0 * mu[i].c00;
        RE(ier, ihp+i1) += m * fe * (r * Dp + DDp);
        RE(ier, ihp-i1) += m * fe * (r * Dm + DDm);
        RE(ier, ihr) += fe * mr * (-r + D);
        RE(ier, ihr+i1) += fe * mr * Dp;
        RE(ier, ihr-i1) += fe * mr * Dm;

        RE(iep, ihp) += fe * mr * (r + D);
        RE(iep, ihp+i1) += fe * mr * Dp;
        RE(iep, ihp-i1) += fe * mr * Dm;
        RE(iep, ihr) += fe * mr2 - k0m * mu[i].c11;

        RH(ihp, ier) += - fh * mr2 + k0m * epsilons[layer][i].c11;
        RH(ihp, iep) += fh * mr * (r + D);
        RH(ihp, iep+i1) += fh * mr * Dp;
        RH(ihp, iep-i1) += fh * mr * Dm;

        RH(ihr, ier) += fh * mr * (-r + D);
        RH(ihr, ier+i1) += fh * mr * Dp;
        RH(ihr, ier-i1) += fh * mr * Dm;
        RH(ihr, iep) += m * fh * (r2 - r * D - DD) - mk0 * epsilons[layer][i].c00;
        RH(ihr, iep+i1) += m * fh * (- r * Dp - DDp);
        RH(ihr, iep-i1) += m * fh * (- r * Dm - DDm);
    }

    // At the end we assume both r derivatives to be equal to 0
    {
        const double r = 1. / raxis->at(N1);
        const double r2 = r * r;
        const double mr = this->m? m * r : 0.;
        const double mr2 = this->m? m * r2 : 0.;

        const dcomplex fe = 1. / (k0 * epsilons[layer][N1].c22),
                       fh = 1. / (k0 * mu[N1].c22);

        const size_t ier = iEr(N1), iep = iEp(N1), ihr = iHr(N1), ihp = iHp(N1);

        RE(ier, ihp) += - fe * mr2 + m * k0 * mu[N1].c00;
        RE(ier, ihr) += - fe * mr2;

        RE(iep, ihp) += fe * mr2;
        RE(iep, ihr) += fe * mr2 - k0m * mu[N1].c11;

        RH(ihp, ier) += - fh * mr2 + k0m * epsilons[layer][N1].c11;
        RH(ihp, iep) += fh * mr2;

        RH(ihr, ier) += - fh * mr2;
        RH(ihr, iep) += fh * mr2 - mk0 * epsilons[layer][N1].c00;
    }

}


void ExpansionCylinders::prepareField()
{
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_LINEAR;
}

void ExpansionCylinders::cleanupField()
{
}

LazyData<Vec<3,dcomplex>> ExpansionCylinders::getField(size_t l,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    const cvector& E, const cvector& H)
{
    size_t N = raxis->size();

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(raxis, plask::make_shared<OnePointAxis>(level->vpos()));

    DataVector<Vec<3,dcomplex>> field(N);

    const double m = this->m;

    if (which_field == FIELD_E) {
        for (size_t i = 0; i != N; ++i) {
            field[i].c0 = E[iEp(i)];
            field[i].c1 = m * E[iEr(i)];
            //TODO add z component
            field[i].c2 = 0;
        }
    } else { // which_field == FIELD_H
        for (size_t i = 0; i != N; ++i) {
            field[i].c0 = H[iHp(i)];
            field[i].c1 = m * H[iHr(i)];
            //TODO add z component
            field[i].c2 = 0;
        }
    }

    return interpolate(src_mesh, field, dest_mesh, field_interpolation,
                       InterpolationFlags(SOLVER->getGeometry(),
                                          (this->m % 2)? InterpolationFlags::Symmetry::PPN : InterpolationFlags::Symmetry::NNP,
                                          InterpolationFlags::Symmetry::NO));
}

double ExpansionCylinders::integratePoyntingVert(const cvector& E, const cvector& H) {
    return 1.;
}

LazyData<Tensor3<dcomplex>> ExpansionCylinders::getMaterialNR(size_t layer,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_NEAREST;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    DataVector<Tensor3<dcomplex>> nrs(epsilons[layer].size());
    for (size_t i = 0; i != nrs.size(); ++i) {
        nrs[i] = epsilons[layer][i].sqrt();
    }

    auto base_mesh = plask::make_shared<RectangularMesh<2>>(SOLVER->mesh,
                                                            plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 2));
    auto src_mesh = base_mesh->getElementMesh();
    // We must convert it to datavector, because base_mesh will be destroyed when we exit this method...
    return DataVector<Tensor3<dcomplex>>(interpolate(src_mesh, nrs, dest_mesh, interp,
                       InterpolationFlags(SOLVER->getGeometry(),
                                          InterpolationFlags::Symmetry::POSITIVE,
                                          InterpolationFlags::Symmetry::NO)));
}


}}} // # namespace plask::optical::slab
