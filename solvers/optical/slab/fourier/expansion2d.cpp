#include <boost/algorithm/clamp.hpp>
using boost::algorithm::clamp;

#include "expansion2d.h"
#include "solver2d.h"
#include "../meshadapter.h"

#define SOLVER static_cast<FourierSolver2D*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionPW2D::ExpansionPW2D(FourierSolver2D* solver): Expansion(solver), initialized(false),
    symmetry(E_UNSPECIFIED), polarization(E_UNSPECIFIED) {}

void ExpansionPW2D::setPolarization(Component pol) {
    if (pol != polarization) {
        if (separated() == (pol != E_UNSPECIFIED))
            solver->clearFields();
        else
            SOLVER->invalidate();
        polarization = pol;
    }
}


void ExpansionPW2D::init()
{
    auto geometry = SOLVER->getGeometry();

    shared_ptr<RegularAxis> xmesh;

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    left = geometry->getChild()->getBoundingBox().lower[0];
    right = geometry->getChild()->getBoundingBox().upper[0];

    size_t refine = SOLVER->refine, M;
    if (refine == 0) refine = 1;

    if (symmetry != E_UNSPECIFIED && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
        throw BadInput(solver->getId(), "Symmetry not allowed for asymmetric structure");

    if (geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN)) {
        if (right <= 0.) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0.) throw BadMesh(SOLVER->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric()) left = -right;
    }

    if (!periodic) {
        // Add PMLs
        if (!symmetric()) left -= SOLVER->pml.size + SOLVER->pml.dist;
        right += SOLVER->pml.size + SOLVER->pml.dist;
    }

    double L;
                                                            // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric()) {                                     //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        L = right - left;                                   //  ^ ^ ^ ^ ^
        N = 2 * SOLVER->getSize() + 1;                      // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nN = 4 * SOLVER->getSize() + 1;
        nM = size_t(round(SOLVER->oversampling * double(nN)));  // N = 3  nN = 5  refine = 4  M = 20
        M = refine * nM;                                        // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        double dx = 0.5 * L * double(refine-1) / double(M);     //  ^ ^ ^ ^
        xmesh = plask::make_shared<RegularAxis>(                // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
                               left-dx, right-dx-L/double(M), M);
    } else {
        L = 2. * right;
        N = SOLVER->getSize() + 1;
        nN = 2 * SOLVER->getSize() + 1;
        nM = size_t(round(SOLVER->oversampling * double(nN)));
        M = refine * nM;                                    // N = 3  nN = 5  refine = 4  M = 20
        if (SOLVER->dct2()) {                               // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
            double dx = 0.25 * L / double(M);               //  ^ ^ ^ ^
            xmesh = plask::make_shared<RegularAxis>(        // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
                               dx, right - dx, M);
        } else {
            size_t nNa = 4 * SOLVER->getSize() + 1;
            double dx = 0.5 * L * double(refine-1) / double(refine*nNa);
            xmesh = plask::make_shared<RegularAxis>(-dx, right+dx, M);
        }
    }

    if (nM < nN) throw BadInput(solver->getId(), "Oversampling cannot be smaller than 1");

    SOLVER->writelog(LOG_DETAIL, "Creating{2}{3} expansion with {0} plane-waves (matrix size: {1})",
                     N, matrixSize(), symmetric()?" symmetric":"", separated()?" separated":"");

    if (symmetric()) SOLVER->writelog(LOG_DETAIL, "Symmetry is {0}", (symmetry== E_TRAN)? "Etran" : "Elong");

    matFFT = FFT::Forward1D(4, int(nM), symmetric()? SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1 : FFT::SYMMETRY_NONE);

    // Compute permeability coefficients
    if (periodic) {
        mag.reset(nN, Tensor2<dcomplex>(0.));
        mag[0].c00 = 1.; mag[0].c11 = 1.; // constant 1
    } else {
        DataVector<Tensor2<dcomplex>> work;
        if (nN != nM) {
            mag.reset(nN);
            work.reset(nM, Tensor2<dcomplex>(0.));
        } else {
            mag.reset(nN, Tensor2<dcomplex>(0.));
            work = mag;
        }
        // Add PMLs
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure width: {0}um)", L);
        double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
        if (symmetric()) pil = 0;
        else pil = std::lower_bound(xmesh->begin(), xmesh->end(), pl) - xmesh->begin();
        pir = std::lower_bound(xmesh->begin(), xmesh->end(), pr) - xmesh->begin();
        for (size_t i = 0; i != nM; ++i) {
            for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
                dcomplex sy = 1.;
                if (j < pil) {
                    double h = (pl - xmesh->at(j)) / SOLVER->pml.size;
                    sy = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
                } else if (j > pir) {
                    double h = (xmesh->at(j) - pr) / SOLVER->pml.size;
                    sy = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
                }
                work[i] += Tensor2<dcomplex>(sy, 1./sy);
            }
            work[i] /= (double)refine;
        }
        // Compute FFT
        FFT::Forward1D(2, int(nM), symmetric()? SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1 : FFT::SYMMETRY_NONE)
            .execute(reinterpret_cast<dcomplex*>(work.data()));
        // Copy data to its final destination
        if (nN != nM) {
            if (symmetric()) {
                std::copy_n(work.begin(), nN, mag.begin());
            } else {
                size_t nn = nN/2;
                std::copy_n(work.begin(), nn+1, mag.begin());
                std::copy_n(work.end()-nn, nn, mag.begin()+nn+1);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = PI / L; bb4 *= bb4;   // (2π/L)² / 4
            for (std::size_t i = 0; i != nN; ++i) {
                int k = int(i); if (!symmetric() && k > int(nN/2)) k -= int(nN);
                mag[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = solver->lcount;
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);

    mesh = plask::make_shared<RectangularMesh<2>>(xmesh, solver->verts, RectangularMesh<2>::ORDER_01);

    initialized = true;
}

void ExpansionPW2D::reset() {
    coeffs.clear();
    initialized = false;
    mesh.reset();
}


void ExpansionPW2D::prepareIntegrals(double lam, double glam) {
    temperature = SafeData<double>(SOLVER->inTemperature(mesh), 300.);
    gain_connected = SOLVER->inGain.hasProvider();
    if (gain_connected) {
        if (isnan(glam)) glam = lam;
        gain = SOLVER->inGain(mesh, glam);
    }
}

void ExpansionPW2D::cleanupIntegrals(double, double) {
    temperature.reset();
    gain.reset();
}


void ExpansionPW2D::layerIntegrals(size_t layer, double lam, double glam)
{
    auto geometry = SOLVER->getGeometry();

    size_t refine = SOLVER->refine;
    if (refine == 0) refine = 1;

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points) in thread {}",
                         layer, solver->lcount, refine * nM, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points)",
                         layer, refine * nM);
    #endif

    if (isnan(lam))
        throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    double factor = 1. / double(refine);
    double maty;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            maty = solver->verts->at(i);
            break;
        }
    }
    double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
    Tensor3<dcomplex> refl, refr;
    if (!periodic) {
        double Tl = 0., Tr = 0., totalw = 0.;
        for (size_t i = 0, vl = pil * solver->verts->size(), vr = pir * solver->verts->size(); i != mesh->vert()->size(); ++vl, ++vr, ++i) {
            if (solver->stack[i] == layer) {
                double w = (i == 0 || i == mesh->vert()->size()-1)? 1e-6 : solver->vbounds->at(i) - solver->vbounds->at(i-1);
                Tl += w * temperature[vl]; Tr += w * temperature[vr]; totalw += w;
            }
        }
        Tl /= totalw; Tr /= totalw;
        {
            OmpLockGuard<OmpNestLock> lock; // this must be declared before `material` to guard its destruction
            auto material = geometry->getMaterial(vec(pl,maty));
            lock = material->lock();
            refl = geometry->getMaterial(vec(pl,maty))->NR(lam, Tl).sqr();
            if (isnan(refl.c00) || isnan(refl.c11) || isnan(refl.c22) || isnan(refl.c01))
                throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                               material->name(), lam, Tl);
        }{
            OmpLockGuard<OmpNestLock> lock; // this must be declared before `material` to guard its destruction
            auto material = geometry->getMaterial(vec(pr,maty));
            lock = material->lock();
            refr = geometry->getMaterial(vec(pr,maty))->NR(lam, Tr).sqr();
            if (isnan(refr.c00) || isnan(refr.c11) || isnan(refr.c22) || isnan(refr.c01))
                throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                               material->name(), lam, Tr);
        }
    }

    // Make space for the result
    DataVector<Tensor3<dcomplex>> work;
    if (nN != nM) {
        coeffs[layer].reset(nN);
        work.reset(nM, Tensor3<dcomplex>(0.));
    } else {
        coeffs[layer].reset(nN, Tensor3<dcomplex>(0.));
        work = coeffs[layer];
    }

    // Average material parameters
    for (size_t i = 0; i != nM; ++i) {
        for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
            double T = 0., W = 0.;
            for (size_t k = 0, v = j * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
                if (solver->stack[k] == layer) {
                    double w = (k == 0 || k == mesh->vert()->size()-1)? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                    T += w * temperature[v]; W += w;
                }
            }
            T /= W;
            Tensor3<dcomplex> nr;
            {
                OmpLockGuard<OmpNestLock> lock; // this must be declared before `material` to guard its destruction
                auto material = geometry->getMaterial(vec(mesh->tran()->at(j),maty));
                lock = material->lock();
                nr = material->NR(lam, T);
                if (isnan(nr.c00) || isnan(nr.c11) || isnan(nr.c22) || isnan(nr.c01))
                    throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K", material->name(), lam, T);
            }
            if (nr.c01 != 0.) {
                if (symmetric()) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
                if (separated()) throw BadInput(solver->getId(), "Single polarization not allowed for structure with non-diagonal NR tensor");
            }
            if (gain_connected && solver->lgained[layer]) {
                auto roles = geometry->getRolesAt(vec(mesh->tran()->at(j),maty));
                if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                    Tensor2<double> g = 0.; W = 0.;
                    for (size_t k = 0, v = j * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
                        if (solver->stack[k] == layer) {
                            double w = (k == 0 || k == mesh->vert()->size()-1)? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k-1);
                            g += w * gain[v]; W += w;
                        }
                    }
                    Tensor2<double> ni = glam * g/W * (0.25e-7/PI);
                    nr.c00.imag(ni.c00); nr.c11.imag(ni.c00); nr.c22.imag(ni.c11); nr.c01.imag(0.);
                }
            }
            nr.sqr_inplace();

            // Add PMLs
            if (!periodic) {
                if (j < pil) {
                    double h = (pl - mesh->tran()->at(j)) / SOLVER->pml.size;
                    dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
                    nr = Tensor3<dcomplex>(refl.c00*sy, refl.c11/sy, refl.c22*sy);
                } else if (j > pir) {
                    double h = (mesh->tran()->at(j) - pr) / SOLVER->pml.size;
                    dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
                    nr = Tensor3<dcomplex>(refr.c00*sy, refr.c11/sy, refr.c22*sy);
                }
            }

            work[i] += Tensor3<dcomplex>(nr.c00, nr.c00/(nr.c00*nr.c11-nr.c01*nr.c01), nr.c22, nr.c01);
        }
        work[i] *= factor;
        if (work[i].c11 != 0. && !isnan(work[i].c11.real()) && !isnan(work[i].c11.imag()))
            work[i].c11 = 1. / work[i].c11; // We were averaging inverses of c11 (xx)
        else work[i].c11 = 0.;
        if (work[i].c22 != 0.)
            work[i].c22 = 1. / work[i].c22; // We need inverse of c22 (yy)
    }

    // Check if the layer is uniform
    if (periodic) {
        diagonals[layer] = true;
        for (size_t i = 1; i != nM; ++i) {
            Tensor3<dcomplex> diff = work[i] - work[0];
            if (!(is_zero(diff.c00) && is_zero(diff.c11) && is_zero(diff.c22) && is_zero(diff.c01))) {
                diagonals[layer] = false;
                break;
            }
        }
    } else
        diagonals[layer] = false;

    if (diagonals[layer]) {
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        if (nN != nM) coeffs[layer][0] = work[0];
        std::fill(coeffs[layer].begin()+1, coeffs[layer].end(), Tensor3<dcomplex>(0.));
    } else {
        // Perform FFT
        matFFT.execute(reinterpret_cast<dcomplex*>(work.data()));
        // Copy result
        if (nN != nM) {
            if (symmetric()) {
                std::copy_n(work.begin(), nN, coeffs[layer].begin());
            } else {
                size_t nn = nN/2;
                std::copy_n(work.begin(), nn+1, coeffs[layer].begin());
                std::copy_n(work.end()-nn, nn, coeffs[layer].begin()+nn+1);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = PI / ((right-left) * (symmetric()? 2. : 1.)); bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN; ++i) {
                int k = int(i); if (!symmetric() && k > int(nN/2)) k -= int(nN);
                coeffs[layer][i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
}


LazyData<Tensor3<dcomplex>> ExpansionPW2D::getMaterialNR(size_t l, const shared_ptr<const LevelsAdapter::Level> &level, InterpolationMethod interp)
{
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        if (!symmetric()) {
            return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this,l,dest_mesh](size_t i)->Tensor3<dcomplex>{
                Tensor3<dcomplex> eps(0.);
                for (int k = -int(nN)/2, end = int(nN+1)/2; k != end; ++k) {
                    size_t j = (k>=0)? k : k + nN;
                    eps += coeffs[l][j] * exp(2*PI * k * I * (dest_mesh->at(i).c0-left) / (right-left));
                }
                eps.c22 = 1. / eps.c22;
                eps.sqrt_inplace();
                return eps;
            });
        } else {
            return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this,l,dest_mesh](size_t i)->Tensor3<dcomplex>{
                Tensor3<dcomplex> eps = coeffs[l][0];
                for (std::size_t k = 1; k != nN; ++k) {
                    eps += 2. * coeffs[l][k] * cos(PI * double(k) * dest_mesh->at(i).c0 / (right-left));
                }
                eps.c22 = 1. / eps.c22;
                eps.sqrt_inplace();
                return eps;
            });
        }
    } else {
        DataVector<Tensor3<dcomplex>> params(symmetric()? nN : nN+1);
        std::copy(coeffs[l].begin(), coeffs[l].end(), params.begin());
        FFT::Backward1D(4, int(nN), symmetric()? SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1 : FFT::SYMMETRY_NONE)
            .execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> cmesh = plask::make_shared<RegularAxis>();
        if (symmetric()) {
            if (SOLVER->dct2()) {
                double dx = 0.5 * right / double(nN);
                cmesh->reset(dx, right-dx, nN);
            } else {
                cmesh->reset(0., right, nN);
            }
        } else {
            cmesh->reset(left, right, nN+1);
            params[nN] = params[0];
        }
        for (Tensor3<dcomplex>& eps: params) {
            eps.c22 = 1. / eps.c22;
            eps.sqrt_inplace();
        }
        auto src_mesh = plask::make_shared<RectangularMesh<2>>(cmesh, plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
        return interpolate(src_mesh, params, dest_mesh, interp,
                           InterpolationFlags(SOLVER->getGeometry(),
                                              symmetric()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                              InterpolationFlags::Symmetry::NO)
                          );
    }
}


void ExpansionPW2D::getMatrices(size_t l, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    dcomplex beta{ this->beta.real(),  this->beta.imag() - SOLVER->getMirrorLosses(this->beta.real()/k0.real()) };

    int order = int(SOLVER->getSize());
    dcomplex f = 1. / k0, k02 = k0*k0;
    double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);

    // Ez represents -Ez

    if (separated()) {
        if (symmetric()) {
            // Separated symmetric()
            std::fill_n(RE.data(), N*N, dcomplex(0.));
            std::fill_n(RH.data(), N*N, dcomplex(0.));
            if (polarization == E_LONG) {                   // Ez & Hx
                for (int i = 0; i <= order; ++i) {
                    double gi = b * double(i);
                    size_t ie = iE(i), ih = iH(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = abs(i-j);   double gj = b * double(j);
                        dcomplex fz = (j < 0 && symmetry == E_TRAN)? -f : f;
                        int aj = abs(j);
                        size_t je = iE(aj), jh = iH(aj);
                        RE(ih, je) += fz * (- gi * gj * imuyy(l,ij) + k02 * epszz(l,ij) );
                        RH(ie, jh) += fz *                            k02 * muxx(l,ij);
                    }
                    // Ugly hack to avoid singularity
                    if (RE(ie, ie) == 0.) RE(ie, ie) = 1e-32;
                    if (RH(ih, ih) == 0.) RH(ih, ih) = 1e-32;
                }
            } else {                                        // Ex & Hz
                for (int i = 0; i <= order; ++i) {
                    double gi = b * double(i);
                    size_t ie = iE(i), ih = iH(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = abs(i-j);   double gj = b * double(j);
                        dcomplex fx = (j < 0 && symmetry == E_LONG)? -f : f;
                        int aj = abs(j);
                        size_t je = iE(aj), jh = iH(aj);
                        RE(ih, je) += fx *                             k02 * epsxx(l,ij);
                        RH(ie, jh) += fx * (- gi * gj * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    }
                    // Ugly hack to avoid singularity
                    if (RE(ie, ie) == 0.) RE(ie, ie) = 1e-32;
                    if (RH(ih, ih) == 0.) RH(ih, ih) = 1e-32;
                }
            }
        } else {
            // Separated asymmetric()
            if (polarization == E_LONG) {                   // Ez & Hx
                for (int i = -order; i <= order; ++i) {
                    dcomplex gi = b * double(i) - ktran;
                    size_t ie = iE(i), ih = iH(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - ktran;
                        size_t je = iE(j), jh = iH(j);
                        RE(ih, je) = f * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
                        RH(ie, jh) = f *                              k02 * muxx(l,ij);
                    }
                    // Ugly hack to avoid singularity
                    if (RE(ie, ie) == 0.) RE(ie, ie) = 1e-32;
                    if (RH(ih, ih) == 0.) RH(ih, ih) = 1e-32;
                }
            } else {                                        // Ex & Hz
                for (int i = -order; i <= order; ++i) {
                    dcomplex gi = b * double(i) - ktran;
                    size_t ie = iE(i), ih = iH(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - ktran;
                        size_t je = iE(j), jh = iH(j);
                        RE(ih, je) = f *                               k02 * epsxx(l,ij);
                        RH(ie, jh) = f * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    }
                    // Ugly hack to avoid singularity
                    if (RE(ie, ie) == 0.) RE(ie, ie) = 1e-32;
                    if (RH(ih, ih) == 0.) RH(ih, ih) = 1e-32;
                }
            }
        }
    } else {
        if (symmetric()) {
            // Full symmetric()
            std::fill_n(RE.data(), 4*N*N, dcomplex(0.));
            std::fill_n(RH.data(), 4*N*N, dcomplex(0.));
            for (int i = 0; i <= order; ++i) {
                double gi = b * double(i);
                size_t iex = iEx(i), iez = iEz(i), ihx = iHx(i), ihz = iHz(i);
                for (int j = -order; j <= order; ++j) {
                    int ij = abs(i-j);   double gj = b * double(j);
                    dcomplex fx = (j < 0 && symmetry == E_LONG)? -f : f;
                    dcomplex fz = (j < 0 && symmetry == E_TRAN)? -f : f;
                    int aj = abs(j);
                    size_t jex = iEx(aj), jez = iEz(aj), jhx = iHx(aj), jhz = iHz(aj);
                    RE(ihz, jex) += fx * (- beta*beta * imuyy(l,ij) + k02 * epsxx(l,ij) );
                    RE(ihx, jex) += fx * (  beta* gi  * imuyy(l,ij)                     );
                    RE(ihz, jez) += fz * (  beta* gj  * imuyy(l,ij)                     );
                    RE(ihx, jez) += fz * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
                    RH(iex, jhz) += fx * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    RH(iez, jhz) += fx * (- beta* gj  * iepsyy(l,ij)                    );
                    RH(iex, jhx) += fz * (- beta* gi  * iepsyy(l,ij)                    );
                    RH(iez, jhx) += fz * (- beta*beta * iepsyy(l,ij) + k02 * muxx(l,ij) );
                }
                // Ugly hack to avoid singularity
                if (RE(iex, iex) == 0.) RE(iex, iex) = 1e-32;
                if (RE(iez, iez) == 0.) RE(iez, iez) = 1e-32;
                if (RH(ihx, ihx) == 0.) RH(ihx, ihx) = 1e-32;
                if (RH(ihz, ihz) == 0.) RH(ihz, ihz) = 1e-32;
            }
        } else {
            // Full asymmetric()
            for (int i = -order; i <= order; ++i) {
                dcomplex gi = b * double(i) - ktran;
                size_t iex = iEx(i), iez = iEz(i), ihx = iHx(i), ihz = iHz(i);
                for (int j = -order; j <= order; ++j) {
                    int ij = i-j;   dcomplex gj = b * double(j) - ktran;
                    size_t jex = iEx(j), jez = iEz(j), jhx = iHx(j), jhz = iHz(j);
                    RE(ihz, jex) = f * (- beta*beta * imuyy(l,ij) + k02 * epsxx(l,ij) );
                    RE(ihx, jex) = f * (  beta* gi  * imuyy(l,ij) - k02 * epszx(l,ij) );
                    RE(ihz, jez) = f * (  beta* gj  * imuyy(l,ij) - k02 * epsxz(l,ij) );
                    RE(ihx, jez) = f * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
                    RH(iex, jhz) = f * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    RH(iez, jhz) = f * (- beta* gj  * iepsyy(l,ij)                    );
                    RH(iex, jhx) = f * (- beta* gi  * iepsyy(l,ij)                    );
                    RH(iez, jhx) = f * (- beta*beta * iepsyy(l,ij) + k02 * muxx(l,ij) );
                }
                // Ugly hack to avoid singularity
                if (RE(iex, iex) == 0.) RE(iex, iex) = 1e-32;
                if (RE(iez, iez) == 0.) RE(iez, iez) = 1e-32;
                if (RH(ihx, ihx) == 0.) RH(ihx, ihx) = 1e-32;
                if (RH(ihz, ihz) == 0.) RH(ihz, ihz) = 1e-32;
            }
        }
    }
}


void ExpansionPW2D::prepareField()
{
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_FOURIER;
    if (symmetric()) {
        field.reset(N);
        if (field_interpolation != INTERPOLATION_FOURIER) {
            Component sym = (which_field == FIELD_E)? symmetry : Component(3-symmetry);
            int df = SOLVER->dct2()? 0 : 4;
            fft_x = FFT::Backward1D(1, N, FFT::Symmetry(sym+df), 3);    // tran
            fft_yz = FFT::Backward1D(1, N, FFT::Symmetry(3-sym+df), 3); // long
        }
    } else {
        field.reset(N + 1);
        if (field_interpolation != INTERPOLATION_FOURIER)
            fft_x = FFT::Backward1D(3, N, FFT::SYMMETRY_NONE);
    }
}

void ExpansionPW2D::cleanupField()
{
    field.reset();
    fft_x = FFT::Backward1D();
    fft_yz = FFT::Backward1D();
}

// TODO fields must be carefully verified

LazyData<Vec<3,dcomplex>> ExpansionPW2D::getField(size_t l, const shared_ptr<const typename LevelsAdapter::Level> &level, const cvector& E, const cvector& H)
{
    Component sym = (which_field == FIELD_E)? symmetry : Component((3-symmetry) % 3);

    dcomplex beta{ this->beta.real(),  this->beta.imag() - SOLVER->getMirrorLosses(this->beta.real()/k0.real()) };

    const int order = int(SOLVER->getSize());
    double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double vpos = level->vpos();

    int dx = (symmetric() && field_interpolation != INTERPOLATION_FOURIER && sym != E_TRAN)? 1 : 0; // 1 for sin expansion of tran component
    int dz = (symmetric() && field_interpolation != INTERPOLATION_FOURIER && sym != E_LONG)? 1 : 0; // 1 for sin expansion of long component

    if (which_field == FIELD_E) {
        if (separated()) {
            if (polarization == E_LONG) {
                for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                    field[iE(i)].tran() = field[iE(i)].vert() = 0.;
                    if (iE(i) != 0 || !dz) field[iE(i)-dz].lon() = - E[iE(i)];
                }
            } else { // polarization == E_TRAN
                for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                    field[iE(i)].lon() = 0.;
                    if (iE(i) != 0 || !dx)
                        field[iE(i)-dx].tran() = E[iE(i)];
                    if (iE(i) != 0 || !dz) {
                        field[iE(i)-dz].vert() = 0.; // beta is equal to 0
                        if (symmetric()) {
                            if (symmetry == E_TRAN) { // symmetry == H_LONG
                                for (int j = -order; j <= order; ++j)
                                    field[iE(i)-dz].vert() += iepsyy(l,abs(i-j)) * b*double(j) * H[iH(abs(j))];
                            } else { // symmetry == H_TRAN
                                for (int j = 1; j <= order; ++j)
                                    field[iE(i)-dz].vert() += (iepsyy(l,abs(i-j)) + iepsyy(l,abs(i+j))) * b*double(j) * H[iH(j)];
                            }
                        } else {
                            for (int j = -order; j <= order; ++j)
                                field[iE(i)-dz].vert() += iepsyy(l,i-j) * (b*double(j)-ktran) * H[iH(j)];
                        }
                        field[iE(i)-dz].vert() /= k0;
                    }
                }
            }
        } else {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                if (iE(i) != 0 || !dx)
                    field[iE(i)-dx].tran() = E[iEx(i)];
                if (iE(i) != 0 || !dz) {
                    field[iE(i)-dz].lon() = - E[iEz(i)];
                    if (symmetric()) {
                        if (symmetry == E_TRAN) { // symmetry = H_LONG
                            field[iE(i)-dz].vert() = 0.; // Hx[0] == 0
                            for (int j = 1; j <= order; ++j)
                                field[iE(i)-dz].vert() -= (iepsyy(l,abs(i-j)) - iepsyy(l,abs(i+j))) * (beta * H[iHx(j)] + b*double(j) * H[iHz(j)]);
                        } else { // symmetry = H_TRAN
                            field[iE(i)-dz].vert() = - iepsyy(l,abs(i)) * beta * H[iHx(0)];
                            for (int j = 1; j <= order; ++j)
                                field[iE(i)-dz].vert() -= (iepsyy(l,abs(i-j)) + iepsyy(l,abs(i+j))) * (beta * H[iHx(j)] + b*double(j) * H[iHz(j)]);
                        }
                    } else {
                        field[iE(i)-dz].vert() = 0.;
                        for (int j = -order; j <= order; ++j)
                            field[iE(i)-dz].vert() -= iepsyy(l,i-j) * (beta * H[iHx(i)] + (b*double(j)-ktran) * H[iHz(j)]);
                    }
                    field[iE(i)-dz].vert() /= k0;
                }
            }
        }
    } else { // which_field == FIELD_H
        if (separated()) {
            if (polarization == E_TRAN) {  // polarization == H_LONG
                for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                    field[iH(i)].tran() = field[iH(i)].vert() = 0.;
                    if (iH(i) != 0 || !dz) field[iH(i)- dz].lon() = H[iH(i)];
                }
            } else {  // polarization == H_TRAN
                for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                    field[iH(i)].lon() = 0.;
                    if (iH(i) != 0 || !dx)
                        field[iH(i)-dx].tran() = H[iH(i)];
                    if (iH(i) != 0 || !dz) {
                        field[iH(i)-dz].vert() = 0.; // beta is equal to 0
                        if (symmetric()) {
                            if (symmetry == E_LONG) {
                                for (int j = -order; j <= order; ++j)
                                    field[iH(i)-dz].vert() -= imuyy(l,abs(i-j)) * b*double(j) * E[iE(abs(j))];
                            } else { // symmetry == E_TRAN
                                for (int j = 1; j <= order; ++j)
                                    field[iH(i)-dz].vert() -= (imuyy(l,abs(i-j)) + imuyy(l,abs(i+j))) * b*double(j) * E[iE(j)];
                            }
                        } else {
                            for (int j = -order; j <= order; ++j)
                                field[iH(i)-dz].vert() -= imuyy(l,i-j) * (b*double(j)-ktran) * E[iE(j)];
                        }
                        field[iH(i)-dz].vert() /= k0;
                    }
                }
            }
        } else {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                if (iH(i) != 0 || !dx)
                    field[iH(i)-dx].tran() = H[iHx(i)];
                if (iH(i) != 0 || !dz) {
                    field[iH(i)-dz].lon() = H[iHz(i)];
                    field[iH(i)-dz].vert() = 0.;
                    if (symmetric()) {
                        if (symmetry == E_LONG) {
                            field[iE(i)-dz].vert() = 0.; // Ex[0] = 0
                            for (int j = 1; j <= order; ++j)
                                field[iE(i)-dz].vert() += (imuyy(l,abs(i-j)) - imuyy(l,abs(i+j))) * (beta * E[iEx(j)] - b*double(j) * E[iEz(j)]);
                        } else { // symmetry == E_TRAN
                            field[iE(i)-dz].vert() = imuyy(l,abs(i)) * beta * E[iEx(0)];
                            for (int j = 1; j <= order; ++j)
                                field[iE(i)-dz].vert() += (imuyy(l,abs(i-j)) + imuyy(l,abs(i+j))) * (beta * E[iEx(j)] - b*double(j) * E[iEz(j)]);
                        }
                    } else {
                        field[iH(i)-dz].vert() = 0.;
                        for (int j = -order; j <= order; ++j)
                            field[iE(i)-dz].vert() += imuyy(l,i-j) * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
                    }
                    field[iH(i)].vert() /= k0;
                }
            }
        }
    }

    if (dx) { field[field.size()-1].tran() = 0.; }
    if (dz) { field[field.size()-1].lon() = 0.; field[field.size()-1].vert() = 0.; }

    if (field_interpolation == INTERPOLATION_FOURIER) {
        DataVector<Vec<3,dcomplex>> result(dest_mesh->size());
        double L = right - left;
        if (!symmetric()) {
            dcomplex B = 2*PI * I / L;
            dcomplex ikx = I * ktran;
            result.reset(dest_mesh->size(), Vec<3,dcomplex>(0.,0.,0.));
            for (int k = -order; k <= order; ++k) {
                size_t j = (k>=0)? k : k + N;
                dcomplex G = B * double(k) - ikx;
                for (size_t i = 0; i != dest_mesh->size(); ++i) {
                    double x = dest_mesh->at(i)[0];
                    if (!periodic) x = clamp(x, left, right);
                    result[i] += field[j] * exp(G * (x-left));
                }
            }
        } else {
            double B = PI / L;
            result.reset(dest_mesh->size());
            for (size_t i = 0; i != dest_mesh->size(); ++i) {
                result[i] = field[0];
                for (int k = 1; k <= order; ++k) {
                    double x = dest_mesh->at(i)[0];
                    if (!periodic) x = clamp(x, -right, right);
                    double cs =  2. * cos(B * k * x);
                    double sn =  2. * sin(B * k * x);
                    if (sym == E_TRAN) {
                        result[i].lon() += field[k].lon() * sn;
                        result[i].tran() += field[k].tran() * cs;
                        result[i].vert() += field[k].vert() * sn;
                    } else {
                        result[i].lon() += field[k].lon() * cs;
                        result[i].tran() += field[k].tran() * sn;
                        result[i].vert() += field[k].vert() * cs;
                    }
                }
            }
        }
        return result;
    } else {
        if (symmetric()) {
            fft_x.execute(&(field.data()->tran()));
            fft_yz.execute(&(field.data()->lon()));
            fft_yz.execute(&(field.data()->vert()));
            double dx = 0.5 * (right-left) / double(N);
            auto src_mesh = plask::make_shared<RectangularMesh<2>>(plask::make_shared<RegularAxis>(left+dx, right-dx, field.size()), plask::make_shared<RegularAxis>(vpos, vpos, 1));
            return interpolate(src_mesh, field, dest_mesh, field_interpolation,
                               InterpolationFlags(SOLVER->getGeometry(),
                                    (sym == E_TRAN)? InterpolationFlags::Symmetry::NPN : InterpolationFlags::Symmetry::PNP,
                                    InterpolationFlags::Symmetry::NO),
                                    false);
        } else {
            fft_x.execute(reinterpret_cast<dcomplex*>(field.data()));
            field[N] = field[0];
            auto src_mesh = plask::make_shared<RectangularMesh<2>>(plask::make_shared<RegularAxis>(left, right, field.size()), plask::make_shared<RegularAxis>(vpos, vpos, 1));
            auto result = interpolate(src_mesh, field, dest_mesh, field_interpolation,
                                      InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::NO),
                                      false).claim();
            dcomplex ikx = I * ktran;
            for (size_t i = 0; i != dest_mesh->size(); ++i)
                result[i] *= exp(- ikx * dest_mesh->at(i).c0);
            return result;
        }
    }
}


double ExpansionPW2D::integratePoyntingVert(const cvector& E, const cvector& H)
{
    double P = 0.;

    const int ord = int(SOLVER->getSize());

    if (separated()) {
        if (symmetric()) {
            for (int i = 0; i <= ord; ++i) {
                P += real(E[iE(i)] * conj(H[iH(i)]));
            }
            P = 2. * P - real(E[iE(0)] * conj(H[iH(0)]));
        } else {
            for (int i = -ord; i <= ord; ++i) {
                P += real(E[iE(i)] * conj(H[iH(i)]));
            }
        }
    } else {
        if (symmetric()) {
            for (int i = 0; i <= ord; ++i) {
                P -= real(E[iEz(i)] * conj(H[iHx(i)]) + E[iEx(i)] * conj(H[iHz(i)]));
            }
            P = 2. * P + real(E[iEz(0)] * conj(H[iHx(0)]) + E[iEx(0)] * conj(H[iHz(0)]));
        } else {
            for (int i = -ord; i <= ord; ++i) {
                P -= real(E[iEz(i)] * conj(H[iHx(i)]) + E[iEx(i)] * conj(H[iHz(i)]));
            }
        }
    }

    double L = SOLVER->geometry->getExtrusion()->getLength();
    if (!isinf(L))
        P *= L * 1e-6;

    return P * (right - left) * 1e-6; // µm² -> m²
}


void ExpansionPW2D::getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix& RE, const cdiagonal& gamma)
{
    size_t nr = Te.rows(), nc = Te.cols();
    std::fill_n(Te.data(), nr*nc, 0.);
    std::fill_n(Te1.data(), nr*nc, 0.);

    if (separated()) {
        for (std::size_t i = 0; i < nc; i++) {
            Te(i,i) = Te1(i,i) = 1.;
        }
    } else {
        // Ensure that for the same gamma E*H [2x2] is diagonal
        assert(nc % 2 == 0);
        size_t n = nc / 2;
        for (std::size_t i = 0; i < n; i++) {
            // Compute Te1 = sqrt(RE)
            // https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
            // but after this normalize columns to 1
            dcomplex a = RE(2*i, 2*i), b = RE(2*i, 2*i+1), c = RE(2*i+1, 2*i), d = RE(2*i+1, 2*i+1);
            dcomplex s = sqrt(a*d - b*c);
            a += s; d += s;
            // Normalize
            s = 1. / sqrt(a*a + b*b); a *= s; b *= s;
            s = 1. / sqrt(c*c + d*d); c *= s; d *= s;
            Te1(2*i, 2*i) = a; Te1(2*i, 2*i+1) = b;
            Te1(2*i+1, 2*i) = c; Te1(2*i+1, 2*i+1) = d;
            // Invert Te1
            s = 1. / (a*d - b*c);
            Te(2*i, 2*i) = s * d; Te(2*i, 2*i+1) = - s * b;
            Te(2*i+1, 2*i) = - s * c; Te(2*i+1, 2*i+1) = s * a;
        }
    }
}

}}} // namespace plask
