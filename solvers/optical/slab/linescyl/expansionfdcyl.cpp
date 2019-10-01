#include "expansionfdcyl.hpp"
#include "solverfdcyl.hpp"

#define SOLVER static_cast<LinesSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionLines::ExpansionLines(LinesSolverCyl* solver): Expansion(solver), m(1),
                                                           initialized(false)
{
}


size_t ExpansionLines::matrixSize() const {
    return 2 * raxis->size() - ((m==1)? 1 : 2);
}


void ExpansionLines::init()
{
    size_t nlayers = solver->lcount;
    diagonals.assign(nlayers, false);
    epsilons.resize(nlayers);

    if (is_zero(SOLVER->mesh->at(0))) {
        raxis = SOLVER->mesh;
    } else {
        shared_ptr<OrderedAxis> oaxis(new OrderedAxis(*SOLVER->mesh));
        oaxis->addPoint(0.);
        raxis = oaxis;
    }

    // Fill-in mu
    size_t n = raxis->size();
    mu.reset(n, Tensor3<dcomplex>(1.));
    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
        auto bbox = SOLVER->geometry->getChild()->getBoundingBox();
        double shift = max(-bbox.lower.c0, bbox.upper.c0) + SOLVER->pml.dist;
        if (raxis->at(raxis->size()-1) < shift + SOLVER->pml.size - 1e-3)
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


void ExpansionLines::reset()
{
    raxis.reset();
    epsilons.clear();
    mu.reset();
    initialized = false;
}



void ExpansionLines::beforeLayersIntegrals(double lam, double glam) {
    auto mesh = plask::make_shared<RectangularMesh<2>>(raxis, solver->verts, RectangularMesh<2>::ORDER_01);
    temperature = SOLVER->inTemperature(mesh);
    gain_connected = SOLVER->inGain.hasProvider();
    if (gain_connected) {
        if (isnan(glam)) glam = lam;
        gain = SOLVER->inGain(mesh, glam);
    }
}

void ExpansionLines::afterLayersIntegrals() {
    temperature.reset();
    gain.reset();
}

void ExpansionLines::layerIntegrals(size_t layer, double lam, double glam) {
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

        Tensor3<dcomplex> eps[2];
        {
            OmpLockGuard<OmpNestLock> lock; // this must be declared before `material` to guard its destruction
            double T = 0., W = 0.;
            for (size_t k = 0, v = ri * solver->verts->size(); k != solver->verts->size(); ++v, ++k) {
                if (solver->stack[k] == layer) {
                    double w = (k == 0 || k == solver->verts->size()-1)?
                            1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                    T += w * temperature[v]; W += w;
                }
            }
            T /= W;
            for (int i = 0; i != 2; ++i) {
                auto material = geometry->getMaterial(vec(r + 1e-3*i-0.5e-3, matz));
                lock = material->lock();
                eps[i] = material->NR(lam, T);
                if (isnan(eps[i].c00) || isnan(eps[i].c11) || isnan(eps[i].c22) || isnan(eps[i].c01))
                    throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                                   material->name(), lam, T);
            }
        }
        if (eps[0].c01 != 0. || eps[1].c01 != 0.)
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
                eps[0].c00.imag(ni.c00); eps[0].c11.imag(ni.c00); eps[0].c22.imag(ni.c11);
                eps[1].c00.imag(ni.c00); eps[1].c11.imag(ni.c00); eps[1].c22.imag(ni.c11);
            }
        }
        eps[0].sqr_inplace();
        eps[1].sqr_inplace();

        epsilons[layer][ri] = Tensor3<dcomplex>(
            0.5 * (eps[0].c00 + eps[1].c00),
            2.0 / (1./eps[0].c11 + 1./eps[1].c11),
            2.0 / (1./eps[0].c22 + 1./eps[1].c22)
        );
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


void ExpansionLines::getMatrices(size_t layer, cmatrix& RH, cmatrix& RE)
{
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    size_t N1 = raxis->size() - 1;

    std::fill(RH.begin(), RH.end(), dcomplex(0.));
    std::fill(RE.begin(), RE.end(), dcomplex(0.));

    const int m2 = m * m;

    //The first element
    if (m == 1) {
//         size_t ier = iEr(0), ihp = iHp(0);
//
//         const dcomplex fe = 1. / (k0 * epsilons[layer][0].c22),
//                        fh = 1. / (k0 * mu[0].c22);
//
//         RH(ier, ihp) += fe * (r * DD) + k0 * r * mu[i].c00;
//         RH(ier, ihp+i1) += fe * (r * DDp);
//         RH(ier, ihr) += fe * (-mr2);
//
//         RE(ihr, ier) += fh * (-mr2);
//         RE(ihr, iep) += fh * (-r * DD) - k0 * r * epsilons[layer][i].c00;
//         RE(ihr, iep+i1) += fh * (-r * DDp);
    }

    for (size_t i = 1; i != N1; ++i) {
        size_t ier = iEr(i), iep = iEp(i), ihr = iHr(i), ihp = iHp(i);

        const double r = 1. / raxis->at(i), rm = (i==1)? 1. : 1. / raxis->at(i-1), rp = raxis->at(i+1);
        const double rsm = 2. / (raxis->at(i) + raxis->at(i-1)), rsp = 2. / (raxis->at(i) + raxis->at(i+1));
        const double mr = m*r, m2r = m2*r, mrm = m*rm, mrp = m*rp;

        const dcomplex fe = 1. / (k0 * epsilons[layer][i].c22),
                       fh = 1. / (k0 * mu[i].c22);

        const double dm = raxis->at(i) - raxis->at(i-1), dp = raxis->at(i+1) - raxis->at(i);
        const double D = (dp-dm) / (dm*dp), Dp = dm/dp / (dm+dp), Dm = - dp/dm / (dm+dp);
        const double DrD = - 2. / (dm*dp) * (rsm*rsp) / r + (dp-dm)*(dp+dm)/(dm*dm*dp*dp) * r,
                     DrDp = 2. / (dm+dp) * (rsm / dp + r / (dp*dp) * (dp-dm)),
                     DrDm = 2. / (dm+dp) * (rsp / dm - r / (dm*dm) * (dp-dm));

        const dcomplex de = Dm / (k0 * epsilons[layer][i-1].c22) +
                            D  / (k0 * epsilons[layer][i].c22) +
                            Dp / (k0 * epsilons[layer][i+1].c22);
        const dcomplex dh = Dm / (k0 * mu[i-1].c22) + D / (k0 * mu[i].c22) + Dp / (k0 * mu[i+1].c22);

        RE(ihr, iep) += fh * DrD + dh * r * D + k0 * r * epsilons[layer][i].c00;
        RE(ihr, iep+i1) += fh * DrDp + dh * r * Dp;
        if (i != 1)
            RE(ihr, iep-i1) += fh * DrDm + dh * r * Dm;
        RE(ihr, ier) += fh * mr * D + dh * mr;
        RE(ihr, ier+i1) += fh * mrp * Dp;
        if (i != 1 || m == 1)
            RE(ihr, ier-i1) += fh * mrm * Dm;

        RE(ihp, iep) -= fh * mr * D;
        RE(ihp, iep+i1) -= fh * mr * Dp;
        if (i != 1)
            RE(ihp, iep-i1) -= fh * mr * Dm;
        RE(ihp, ier) += - fh * m2r + k0 * raxis->at(i) * epsilons[layer][i].c11;

        RH(iep, ihr) += - fe * m2r + k0 * raxis->at(i) * mu[i].c11;
        RH(iep, ihp) += fe * mr * D;
        RH(iep, ihp+i1) += fe * mr * Dp;
        if (i != 1)
            RH(iep, ihp-i1) -= fe * mr * Dm;

        RH(ier, ihr) += fe * mr * D + de * mr;
        RH(ier, ihr+i1) += fe * mrp * Dp;
        if (i != 1 || m == 1)
            RH(ier, ihr-i1) += fe * mrm * Dm;
        RH(ier, ihp) += fe * DrD + de * r * D + k0 * r * mu[i].c00;
        RH(ier, ihp+i1) += fe * DrDp + de * r * Dp;
        if (i != 1)
            RH(ier, ihp-i1) += fe * DrDm + de * r * Dm;
    }

    // At the end we assume second r derivative to be equal to 0
    {
        size_t ier = iEr(N1), iep = iEp(N1), ihr = iHr(N1), ihp = iHp(N1);

        const double r = 1. / raxis->at(N1), rm = raxis->at(N1-1);
        const double rsm = 2. / (raxis->at(N1) + raxis->at(N1-1));
        const double mr = m*r, m2r = m2*r, mrm = m*rm;

        const dcomplex fe = 1. / (k0 * epsilons[layer][N1].c22),
                       fh = 1. / (k0 * mu[N1].c22);

        const double D = 1 / rsm, Dm = - D;  // TODO: Verify this!!

        RE(ihr, iep) += k0 * r * epsilons[layer][N1].c00;
        RE(ihr, ier) += fh * mr * D;
        RE(ihr, ier-i1) += fh * mrm * Dm;

        RE(ihp, iep) -= fh * mr * D;
        RE(ihp, iep-i1) -= fh * mr * Dm;
        RE(ihp, ier) += - fh * m2r + k0 * raxis->at(N1) * epsilons[layer][N1].c11;

        RH(iep, ihr) += - fe * m2r + k0 * raxis->at(N1) * mu[N1].c11;
        RH(iep, ihp) += fe * mr * D;
        RH(iep, ihp-i1) -= fe * mr * Dm;

        RH(ier, ihr) += fe * mr * D;
        RH(ier, ihr-i1) += fe * mrm * Dm;
        RH(ier, ihp) += k0 * r * mu[N1].c00;
    }

}


void ExpansionLines::prepareField()
{
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_LINEAR;
}

void ExpansionLines::cleanupField()
{
}

LazyData<Vec<3,dcomplex>> ExpansionLines::getField(size_t l,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    const cvector& E, const cvector& H)
{
    size_t N = raxis->size();

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(raxis, plask::make_shared<OnePointAxis>(level->vpos()));

    DataVector<Vec<3,dcomplex>> field(N);

    if (which_field == FIELD_E) {
        if (m == 1) {
            field[0].c0 =   E[iEr(0)];
            field[0].c1 = - E[iEr(0)];
        } else {
            field[0].c0 = field[0].c1 = 0.;
        }
        field[0].c2 = 0;
        for (size_t i = 1; i != N; ++i) {
            field[i].c0 = E[iEp(i)] / SOLVER->mesh->at(i);
            field[i].c1 = E[iEr(i)];
            //TODO add z component
            field[i].c2 = 0;
        }
    } else { // which_field == FIELD_H
        if (m == 1) {
            field[0].c0 =   H[iHr(0)];
            field[0].c1 = - H[iHr(0)];
        } else {
            field[0].c0 = field[0].c1 = 0.;
        }
        field[0].c2 = 0;
        for (size_t i = 1; i != N; ++i) {
            field[i].c0 = H[iHp(i)] / SOLVER->mesh->at(i);
            field[i].c1 = - H[iHr(i)];
            //TODO add z component
            field[i].c2 = 0;
        }
    }

    return interpolate(src_mesh, field, dest_mesh, field_interpolation,
                       InterpolationFlags(SOLVER->getGeometry(),
                                          (this->m % 2)? InterpolationFlags::Symmetry::PPN : InterpolationFlags::Symmetry::NNP,
                                          InterpolationFlags::Symmetry::NO));
}

double ExpansionLines::integratePoyntingVert(const cvector& E, const cvector& H) {
    return 1.;
}

LazyData<Tensor3<dcomplex>> ExpansionLines::getMaterialNR(size_t layer,
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

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(raxis,
                                                           plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    // We must convert it to datavector, because base_mesh will be destroyed when we exit this method...
    return DataVector<Tensor3<dcomplex>>(interpolate(src_mesh, nrs, dest_mesh, interp,
                       InterpolationFlags(SOLVER->getGeometry(),
                                          InterpolationFlags::Symmetry::POSITIVE,
                                          InterpolationFlags::Symmetry::NO)));
}


}}} // # namespace plask::optical::slab
