#include <boost/algorithm/clamp.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/adaptor/transformed.hpp>
using boost::algorithm::clamp;

#include "expansion2d.hpp"
#include "solver2d.hpp"
#include "../meshadapter.hpp"

#define SOLVER static_cast<FourierSolver2D*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionPW2D::ExpansionPW2D(FourierSolver2D* solver): Expansion(solver), initialized(false),
    symmetry(E_UNSPECIFIED), polarization(E_UNSPECIFIED) {}

void ExpansionPW2D::setPolarization(Component pol) {
    if (pol != polarization) {
        solver->clearFields();
        if (!periodic && polarization == E_TRAN) {
            polarization = pol;
            if (initialized) {
                reset();
                init();
            }
            solver->recompute_integrals = true;
        } else if (polarization != E_UNSPECIFIED) {
            polarization = pol;
            solver->recompute_integrals = true;
        } else {
            polarization = pol;
        }
    }
}


void ExpansionPW2D::init()
{
    auto geometry = SOLVER->getGeometry();

    shared_ptr<MeshAxis> xmesh;

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    auto bbox = geometry->getChild()->getBoundingBox();
    left = bbox.lower[0];
    right = bbox.upper[0];

    size_t refine = SOLVER->refine, nrN;
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

    if (SOLVER->ftt == FourierSolver2D::FOURIER_ANALYTIC) {
        if (!SOLVER->mesh) SOLVER->setSimpleMesh();
        if (SOLVER->mesh->size() < 2) throw BadInput(SOLVER->getId(), "Mesh needs at least two points");
        if (!geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN) || SOLVER->mesh->at(0) < 0.) {
            original_mesh = SOLVER->mesh;
        } else {
            shared_ptr<OrderedAxis> new_mesh = make_shared<OrderedAxis>(*SOLVER->mesh);
            original_mesh = new_mesh;
            auto negate = [](double x) { return -x; };
            auto transformed = (*original_mesh) | boost::adaptors::reversed | boost::adaptors::transformed(negate);
            new_mesh->addOrderedPoints(transformed.begin(), transformed.end(), new_mesh->size());
        }
        if (!is_zero(original_mesh->at(0) - (symmetric()? -right : left)))
            throw BadInput(SOLVER->getId(), "First mesh point ({}) must match left geometry boundary ({})",
                           original_mesh->at(0), symmetric()? -right : left);
        if (!is_zero(original_mesh->at(original_mesh->size()-1) - right))
            throw BadInput(SOLVER->getId(), "Last mesh point ({}) must match right geometry boundary ({})",
                           original_mesh->at(original_mesh->size()-1), right);
    }

    if (!periodic) {
        // Add PMLs
        if (!symmetric()) left -= SOLVER->pml.size + SOLVER->pml.dist;
        right += SOLVER->pml.size + SOLVER->pml.dist;
    }

    double L;
                                                                // N = 3  nN = 5  refine = 5  nrN = 25
    if (!symmetric()) {                                         //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        L = right - left;                                       //  ^ ^ ^ ^ ^
        N = 2 * SOLVER->getSize() + 1;                          // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nN = 4 * SOLVER->getSize() + 1;                             // N = 3  nN = 5  refine = 4  nrN = 20
        nrN = refine * nN;                                          // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        double dx = 0.5 * L * double(refine-1) / double(nrN);       //  ^ ^ ^ ^
        if (SOLVER->ftt == FourierSolver2D::FOURIER_DISCRETE)       // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
            xmesh = plask::make_shared<RegularAxis>(left-dx, right-dx-L/double(nrN), nrN);
    } else {
        L = 2. * right;
        N = SOLVER->getSize() + 1;
        nN = 2 * SOLVER->getSize() + 1;
        nrN = refine * nN;                                          // N = 3  nN = 5  refine = 4  nrN = 20
        if (SOLVER->dct2()) {                                       // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
            double dx = 0.25 * L / double(nrN);                     //  ^ ^ ^ ^
            if (SOLVER->ftt == FourierSolver2D::FOURIER_DISCRETE)   // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
                xmesh = plask::make_shared<RegularAxis>(dx, right-dx, nrN);
        } else {
            size_t nNa = 4 * SOLVER->getSize() + 1;
            double dx = 0.5 * L * double(refine-1) / double(refine*nNa);
            if (SOLVER->ftt == FourierSolver2D::FOURIER_DISCRETE)
                xmesh = plask::make_shared<RegularAxis>(-dx, right+dx, nrN);
        }
    }
    size_t M = matrixSize();

    SOLVER->writelog(LOG_DETAIL, "Creating{2}{3} expansion with {0} plane-waves (matrix size: {1})",
                     N, M, symmetric()?" symmetric":"", separated()?" separated":"");

    if (symmetric()) SOLVER->writelog(LOG_DETAIL, "Symmetry is {0}", (symmetry== E_TRAN)? "Etran" : "Elong");

    if (SOLVER->ftt == FourierSolver2D::FOURIER_DISCRETE) {
        matFFT = FFT::Forward1D(1, int(nN),
                                symmetric()?
                                    SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1 :
                                    FFT::SYMMETRY_NONE);
    } else {
        if (!periodic) {
            xmesh = plask::make_shared<OrderedAxis>(*original_mesh->getMidpointAxis());
            static_pointer_cast<OrderedAxis>(xmesh)->addPoint(SOLVER->mesh->at(0) - 2.*OrderedAxis::MIN_DISTANCE);
            static_pointer_cast<OrderedAxis>(xmesh)->addPoint(SOLVER->mesh->at(SOLVER->mesh->size()-1) + 2.*OrderedAxis::MIN_DISTANCE);
        } else
            xmesh = original_mesh->getMidpointAxis();
    }

    // Compute permeability coefficients
    if (periodic) {
        mag.reset(nN, 0.);
        mag[0] = 1.;
        if (polarization != E_TRAN) {
            rmag.reset(nN, 0.);
            rmag[0] = 1.;
        }
    } else {
        if (SOLVER->ftt == FourierSolver2D::FOURIER_DISCRETE) {
            mag.reset(nN, 0.);
            if (polarization != E_TRAN) rmag.reset(nN, 0.);
            // Add PMLs
            SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure width: {0}um)", L);
            double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
            if (symmetric()) pil = 0;
            else pil = std::lower_bound(xmesh->begin(), xmesh->end(), pl) - xmesh->begin();
            pir = std::lower_bound(xmesh->begin(), xmesh->end(), pr) - xmesh->begin();
            for (size_t i = 0; i != nN; ++i) {
                for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
                    dcomplex sy = 1.;
                    if (j < pil) {
                        double h = (pl - xmesh->at(j)) / SOLVER->pml.size;
                        sy = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
                    } else if (j > pir) {
                        double h = (xmesh->at(j) - pr) / SOLVER->pml.size;
                        sy = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
                    }
                    mag[i] += sy;
                    if (polarization != E_TRAN) rmag[i] += 1. / sy;
                }
                mag[i] /= (double)refine;
                if (polarization != E_TRAN) rmag[i] /= (double)refine;
            }
            // Compute FFT
            FFT::Forward1D fft(1, int(nN),
                           symmetric()?
                               SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1 :
                               FFT::SYMMETRY_NONE);
            fft.execute(mag.data());
            if (polarization != E_TRAN) fft.execute(rmag.data());
        } else {
            throw NotImplemented(SOLVER->getId(), "Analytic Fourier transform for non-periodic structure");  //TODO
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = PI / L; bb4 *= bb4;   // (2π/L)² / 4
            for (std::size_t i = 0; i != nN; ++i) {
                int k = int(i); if (!symmetric() && k > int(nN/2)) k -= int(nN);
                mag[i] *= exp(-SOLVER->smooth * bb4 * k * k);
                if (polarization != E_TRAN) rmag[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
        TempMatrix temp = getTempMatrix();
        cmatrix work(temp);
        cmatrix workij(N, N, work.data());
        make_permeability_matrices(workij);
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = solver->lcount;
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);
    coeff_matrices.resize(nlayers);

    mesh = plask::make_shared<RectangularMesh<2>>(xmesh, solver->verts, RectangularMesh<2>::ORDER_01);

    initialized = true;
}

void ExpansionPW2D::reset() {
    coeffs.clear();
    coeff_matrices.clear();
    coeff_matrix_mxx.reset();
    coeff_matrix_rmyy.reset();
    initialized = false;
    mesh.reset();
    mag.reset();
    rmag.reset();
    temporary.reset();
}


void ExpansionPW2D::beforeLayersIntegrals(double lam, double glam) {
    SOLVER->prepareExpansionIntegrals(this, mesh, lam, glam);
}


void ExpansionPW2D::layerIntegrals(size_t layer, double lam, double glam)
{
    auto geometry = SOLVER->getGeometry();

    double maty;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            maty = solver->verts->at(i);
            break;
        }
    }

    double L = (right-left) * (symmetric()? 2. : 1.);

    bool epsilon_isotropic = true, epsilon_diagonal = true;

    if (SOLVER->ftt == FourierSolver2D::FOURIER_DISCRETE) {
        size_t refine = SOLVER->refine;
        if (refine == 0) refine = 1;

        #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
            SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points) in thread {}",
                            layer, solver->lcount, refine * nN, omp_get_thread_num());
        #else
            SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points)",
                            layer, solver->lcount, refine * nN);
        #endif

        if (isnan(lam))
            throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

        if (gain_connected && solver->lgained[layer]) {
            SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
            if (isnan(glam)) glam = lam;
        }

        double factor = 1. / double(refine);

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
        if (polarization == E_LONG) {
            coeffs[layer].zz.reset(nN, 0.);
        } else {
            coeffs[layer].rxx.reset(nN, 0.);
            coeffs[layer].yy.reset(nN, 0.);
        }

        // Average material parameters
        for (size_t i = 0; i != nN; ++i) {
            for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
                Tensor3<dcomplex> eps = getEpsilon(geometry, layer, maty, lam, glam, j);

                // Add PMLs
                if (!periodic) {
                    if (j < pil) {
                        double h = (pl - mesh->tran()->at(j)) / SOLVER->pml.size;
                        dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
                        eps = Tensor3<dcomplex>(refl.c00*sy, refl.c11/sy, refl.c22*sy);
                    } else if (j > pir) {
                        double h = (mesh->tran()->at(j) - pr) / SOLVER->pml.size;
                        dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
                        eps = Tensor3<dcomplex>(refr.c00*sy, refr.c11/sy, refr.c22*sy);
                    }
                }

                if (polarization == E_LONG) {
                    if (eps.c01 != 0.)
                        throw BadInput(solver->getId(), "Polarization can be specified only for diagonal refractive index tensor (NR)");
                    coeffs[layer].zz[i] += eps.c00;
                } else if (polarization == E_TRAN) {
                    if (eps.c01 != 0.)
                        throw BadInput(solver->getId(), "Polarization can be specified only for diagonal refractive index tensor (NR)");
                    coeffs[layer].rxx[i] += 1./eps.c11;
                    coeffs[layer].yy[i] += eps.c22;
                } else {
                    dcomplex rm;
                    bool nd = false;
                    if (eps.c01 != 0.) {
                        if (epsilon_diagonal) {
                            if (symmetric())
                                throw BadInput(solver->getId(), "Symmetry can be specified only for diagonal refractive index tensor (NR)");
                            coeffs[layer].zx.reset(nN, 0.);
                            epsilon_diagonal = false;
                        }
                        nd = true;
                        rm = 1. / (eps.c00*eps.c11 - eps.c01.real()*eps.c01.real() - eps.c01.imag()*eps.c01.imag());
                        coeffs[layer].zx[i] += eps.c01;
                    };
                    if (eps.c00 != eps.c22 || eps.c11 != eps.c22 || !epsilon_isotropic) {
                        if (epsilon_isotropic) {
                            coeffs[layer].zz = coeffs[layer].yy.copy();
                            epsilon_isotropic = false;
                        }
                        coeffs[layer].zz[i] += eps.c00;
                    }
                    coeffs[layer].rxx[i] += nd? rm*eps.c00 : 1./eps.c11;

                    coeffs[layer].yy[i] += eps.c22;
                }
            }

            if (polarization == E_LONG) {
                coeffs[layer].zz[i] *= factor;
            } else {
                coeffs[layer].rxx[i] *= factor;
                coeffs[layer].yy[i] *= factor;
                if (!epsilon_isotropic) {
                    coeffs[layer].zz[i] *= factor;
                }
                if (!epsilon_diagonal) {
                    coeffs[layer].zx[i] *= factor;
                }
            }
        }

        // Check if the layer is uniform
        if (periodic) {
            diagonals[layer] = true;
            if (polarization == E_LONG) {
                for (size_t i = 1; i != nN; ++i)
                    if (!is_zero(coeffs[layer].zz[i] - coeffs[layer].zz[0])) {
                        diagonals[layer] = false; break;
                    }
            } else {
                if (epsilon_isotropic) {
                    for (size_t i = 1; i != nN; ++i)
                        if (!is_zero(coeffs[layer].yy[i] - coeffs[layer].yy[0])) {
                            diagonals[layer] = false; break;
                        }
                } else {
                    for (size_t i = 1; i != nN; ++i)
                        if (!(is_zero(coeffs[layer].zz[i] - coeffs[layer].zz[0]) && is_zero(coeffs[layer].rxx[i] - coeffs[layer].rxx[0]) && is_zero(coeffs[layer].yy[i] - coeffs[layer].yy[0]))) {
                            diagonals[layer] = false; break;
                        }
                }
                if (!epsilon_diagonal) {
                    for (size_t i = 0; i != nN; ++i)
                        if (!is_zero(coeffs[layer].zx[i])) {
                            diagonals[layer] = false; break;
                        }
                }
            }
        } else {
            diagonals[layer] = false;
        }

        if (diagonals[layer]) {
            SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
            size_t n1 = nN - 1;
            if (polarization == E_LONG) {
                std::fill_n(coeffs[layer].zz.data()+1, n1, 0.);
            } else {
                std::fill_n(coeffs[layer].rxx.data()+1, n1, 0.);
                std::fill_n(coeffs[layer].yy.data()+1, n1, 0.);
                if (epsilon_isotropic) {
                    if (polarization != E_TRAN) {
                        coeffs[layer].zz = coeffs[layer].yy;
                    }
                } else {
                    std::fill_n(coeffs[layer].zz.data()+1, n1, 0.);
                }
                if (!epsilon_diagonal) {
                    std::fill_n(coeffs[layer].zx.data()+1, n1, 0.);
                }
            }
        } else {
            // Perform FFT
            if (polarization == E_LONG) {
                matFFT.execute(coeffs[layer].zz.data());
            } else {
                matFFT.execute(coeffs[layer].rxx.data());
                matFFT.execute(coeffs[layer].yy.data());
                if (epsilon_isotropic) {
                    if (polarization != E_TRAN) {
                        coeffs[layer].zz = coeffs[layer].yy;
                    }
                } else {
                    matFFT.execute(coeffs[layer].zz.data());
                }
                if (!epsilon_diagonal) {
                    matFFT.execute(coeffs[layer].zx.data());
                }
            }
        }

    } else {
        #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
            SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} in thread {}",
                            layer, solver->lcount, omp_get_thread_num());
        #else
            SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{}", layer, solver->lcount);
        #endif

        if (isnan(lam))
            throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

        if (gain_connected && solver->lgained[layer]) {
            SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
            if (isnan(glam)) glam = lam;
        }

        size_t mn = mesh->tran()->size();
        const Tensor3<dcomplex> eps0 = getEpsilon(geometry, layer, maty, lam, glam, mn-1);
        bool nd = eps0.c01 != 0.;
        dcomplex rm;
        if (nd) {
            if (polarization != E_UNSPECIFIED)
                throw BadInput(solver->getId(), "Polarization can be specified only for diagonal refractive index tensor (NR)");
            rm = 1. / (eps0.c00*eps0.c11 - eps0.c01.real()*eps0.c01.real() - eps0.c01.imag()*eps0.c01.imag());
        }
        const Tensor3<dcomplex> reps0 = nd? Tensor3<dcomplex>(rm*eps0.c11, rm*eps0.c00, 1./eps0.c22, -rm*eps0.c01) :
                                            Tensor3<dcomplex>(1./eps0.c00, 1./eps0.c11, 1./eps0.c22);

        if (polarization == E_LONG) {
            coeffs[layer].zz.reset(nN, 0.); coeffs[layer].zz[0] = eps0.c00;
        } else {
            coeffs[layer].rxx.reset(nN, 0.); coeffs[layer].rxx[0] = reps0.c11;
            coeffs[layer].yy.reset(nN, 0.); coeffs[layer].yy[0] = eps0.c22;
            if (polarization != E_TRAN) {
                if (eps0.c00 != eps0.c22 || eps0.c11 != eps0.c22) {
                    coeffs[layer].zz.reset(nN, 0.); coeffs[layer].zz[0] = eps0.c00;
                } else {
                    coeffs[layer].zz = coeffs[layer].yy;
                }
            }
        }

        diagonals[layer] = true;

        Tensor3<dcomplex> eps = getEpsilon(geometry, layer, maty, lam, glam, 0), reps;

        double l, r = 0.;
        const ptrdiff_t di = (mesh->tran()->size() == original_mesh->size()+1)? 1 : 0;
        const int start = symmetric()? 0 : -int(nN)/2, end = symmetric()? nN : int(nN+1)/2;
        const double b = 2*PI / L;
        for (size_t i = 1; i < mn; ++i) {
            Tensor3<dcomplex> eps1 = getEpsilon(geometry, layer, maty, lam, glam, i);
            if (!eps1.equals(eps)) {
                nd = eps.c01 != 0.;
                if (nd) {
                    if (polarization != E_UNSPECIFIED)
                        throw BadInput(solver->getId(), "Polarization can be specified only for diagonal refractive index tensor (NR)");
                    rm = 1. / (eps.c00*eps.c11 - eps.c01.real()*eps.c01.real() - eps.c01.imag()*eps.c01.imag());
                    Tensor3<dcomplex>(rm*eps.c11, rm*eps.c00, 1./eps.c22, -rm*eps.c01);
                } else
                    reps = Tensor3<dcomplex>(1./eps.c00, 1./eps.c11, 1./eps.c22);
                diagonals[layer] = false;
                l = r;
                r = original_mesh->at(i-di) - left;
                if (polarization == E_LONG) {
                    add_coeffs(start, end, b, l, r, coeffs[layer].zz, eps.c00 - eps0.c00);
                } else if (polarization == E_TRAN) {
                    add_coeffs(start, end, b, l, r, coeffs[layer].rxx, 1./eps.c11 - reps0.c11);
                    add_coeffs(start, end, b, l, r, coeffs[layer].yy, eps.c22 - eps0.c22);
                } else {
                    if (eps.c00 != eps.c22 || eps.c11 != eps.c22 || !epsilon_isotropic) {
                        if (epsilon_isotropic) {
                            coeffs[layer].zz = coeffs[layer].yy.copy();
                            epsilon_isotropic = false;
                        }
                        add_coeffs(start, end, b, l, r, coeffs[layer].zz, eps.c00 - eps0.c00);
                    }
                    add_coeffs(start, end, b, l, r, coeffs[layer].rxx, reps.c11 - reps0.c11);
                    add_coeffs(start, end, b, l, r, coeffs[layer].yy, eps.c22 - eps0.c22);
                    if (eps.c01 != 0.) {
                        if (epsilon_diagonal) {
                            if (symmetric())
                                throw BadInput(solver->getId(), "Symmetry can be specified only for diagonal refractive index tensor (NR)");
                            coeffs[layer].zx.reset(nN, 0.);
                            epsilon_diagonal = false;
                        }
                        add_coeffs(start, end, b, l, r, coeffs[layer].zx, eps.c01 - eps0.c01);
                    }
                }
                eps = eps1;
            }
        }
        //TODO Add PMLs
    }
    // Smooth coefficients
    if (!diagonals[layer] && SOLVER->smooth) {
        double bb4 = PI / L; bb4 *= bb4;   // (2π/L)² / 4
        for (size_t i = 0; i != nN; ++i) {
            int k = int(i); if (!symmetric() && k > int(nN/2)) k -= int(nN);
            double s = exp(-SOLVER->smooth * bb4 * k * k);
            coeffs[layer].yy[i] *= s;
            if (polarization == E_LONG) {
                if (!epsilon_isotropic) coeffs[layer].zz[i] *= s;
            } else {
                coeffs[layer].rxx[i] *= s;
                if (!epsilon_isotropic) {
                    coeffs[layer].zz[i] *= s;
                }
                if (!epsilon_diagonal) {
                    coeffs[layer].zx[i] *= s;
                }
            }
        }
    }

    //TODO there is no need to do all the above if only symmetry changes (just do the code below)

    // Compute necessary inverses of Toeplitz matrices
    TempMatrix temp = getTempMatrix();
    cmatrix work(N, N, temp.data());

    const int order = int(SOLVER->getSize());

    if (polarization != E_LONG) {
        if (symmetric()) {
            // Full symmetric()
            const bool sel = symmetry == E_LONG;

            for (int i = 0; i <= order; ++i)
                work(i,0) = epsyy(layer,i);
            for (int j = 1; j <= order; ++j)
                for (int i = 0; i <= order; ++i)
                    work(i,j) = sel? epsyy(layer,abs(i-j)) + epsyy(layer,i+j) : epsyy(layer,abs(i-j)) - epsyy(layer,i+j);
            coeff_matrices[layer].reyy.reset(N, N);
            make_unit_matrix(coeff_matrices[layer].reyy);
            invmult(work, coeff_matrices[layer].reyy);

            for (int i = 0; i <= order; ++i)
                work(i,0) = repsxx(layer,i);
            for (int j = 1; j <= order; ++j)
                for (int i = 0; i <= order; ++i)
                    work(i,j) = sel? repsxx(layer,abs(i-j)) - repsxx(layer,i+j) : repsxx(layer,abs(i-j)) + repsxx(layer,i+j);
            coeff_matrices[layer].exx.reset(N, N);
            make_unit_matrix(coeff_matrices[layer].exx);
            invmult(work, coeff_matrices[layer].exx);
        } else {
            for (int j = -order; j <= order; ++j) {
                const size_t jt = iEH(j);
                for (int i = -order; i <= order; ++i) {
                    const size_t it = iEH(i);
                    work(it,jt) = epsyy(layer,i-j);
                }
            }
            coeff_matrices[layer].reyy.reset(N, N);
            make_unit_matrix(coeff_matrices[layer].reyy);
            invmult(work, coeff_matrices[layer].reyy);

            for (int j = -order; j <= order; ++j) {
                const size_t jt = iEH(j);
                for (int i = -order; i <= order; ++i) {
                    const size_t it = iEH(i);
                    work(it, jt) = repsxx(layer,i-j);
                }
            }
            coeff_matrices[layer].exx.reset(N, N);
            make_unit_matrix(coeff_matrices[layer].exx);
            invmult(work, coeff_matrices[layer].exx);
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
                    dcomplex ff = exp(2*PI * k * I * (dest_mesh->at(i).c0-left) / (right-left));
                    switch (polarization) {
                        case E_LONG:
                            eps.c00 += coeffs[l].zz[j] * ff;
                            break;
                        case E_TRAN:
                            eps.c11 += coeffs[l].rxx[j] * ff;
                            eps.c22 += coeffs[l].yy[j] * ff;
                            break;
                        case E_UNSPECIFIED:
                            eps.c00 += coeffs[l].zz[j] * ff;
                            eps.c11 += coeffs[l].rxx[j] * ff;
                            eps.c22 += coeffs[l].yy[j] * ff;
                            if (coeffs[l].zx) eps.c01 += coeffs[l].zx[k] * ff;
                            break;
                    }
                }
                switch (polarization) {
                    case E_LONG:
                        eps.c22 = eps.c11 = eps.c00;
                        break;
                    case E_TRAN:
                        eps.c00 = eps.c22;
                    case E_UNSPECIFIED:
                        eps.c11 = 1. / eps.c11;
                }
                eps.sqrt_inplace();
                return eps;
            });
        } else {
            return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this,l,dest_mesh](size_t i)->Tensor3<dcomplex>{
                Tensor3<dcomplex> eps(0.);
                for (std::size_t k = 0; k != nN; ++k) {
                    dcomplex ff = (k? 2. : 1.) * cos(PI * double(k) * dest_mesh->at(i).c0 / (right-left));
                    switch (polarization) {
                        case E_LONG:
                            eps.c00 += coeffs[l].zz[k] * ff;
                            break;
                        case E_TRAN:
                            eps.c11 += coeffs[l].rxx[k] * ff;
                            eps.c22 += coeffs[l].yy[k] * ff;
                            break;
                        case E_UNSPECIFIED:
                            eps.c00 += coeffs[l].zz[k] * ff;
                            eps.c11 += coeffs[l].rxx[k] * ff;
                            eps.c22 += coeffs[l].yy[k] * ff;
                            break;
                    }
                }
                switch (polarization) {
                    case E_LONG:
                        eps.c22 = eps.c11 = eps.c00;
                        break;
                    case E_TRAN:
                        eps.c00 = eps.c22;
                    case E_UNSPECIFIED:
                        eps.c11 = 1. / eps.c11;
                }
                eps.sqrt_inplace();
                return eps;
            });
        }
    } else {
        DataVector<Tensor3<dcomplex>> params(symmetric()? nN : nN+1);
        FFT::Backward1D fft(1, int(nN), symmetric()? SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1 : FFT::SYMMETRY_NONE, 4);
        if (symmetry == E_LONG) {
            for (size_t i = 0; i != nN; ++i) params[i].c00 = coeffs[l].zz[i];
            fft.execute(reinterpret_cast<dcomplex*>(params.data()));
            for (Tensor3<dcomplex>& eps: params) {
                eps.c22 = eps.c11 = eps.c00;
                eps.sqrt_inplace();
            }
        } else {
            for (size_t i = 0; i != nN; ++i) {
                params[i].c11 = coeffs[l].rxx[i];
                params[i].c22 = coeffs[l].yy[i];
            }
            fft.execute(reinterpret_cast<dcomplex*>(params.data())+1);
            fft.execute(reinterpret_cast<dcomplex*>(params.data())+2);
            if (coeffs[l].zx) {
                for (size_t i = 0; i != nN; ++i) params[i].c01 = coeffs[l].zx[i];
                fft.execute(reinterpret_cast<dcomplex*>(params.data())+3);
            } else {
                for (size_t i = 0; i != nN; ++i) params[i].c01 = 0.;
            }
            if (symmetry == E_TRAN || coeffs[l].zz.data() == coeffs[l].yy.data()) {
                for (Tensor3<dcomplex>& eps: params) {
                    eps.c00 = eps.c22;
                    eps.c11 = 1. / eps.c11;
                    eps.sqrt_inplace();
                }
            } else {
                for (size_t i = 0; i != nN; ++i) params[i].c00 = coeffs[l].zz[i];
                fft.execute(reinterpret_cast<dcomplex*>(params.data()));
                for (Tensor3<dcomplex>& eps: params) {
                    eps.c11 = 1. / eps.c11;
                    eps.sqrt_inplace();
                }
            }
        }
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
        auto src_mesh = plask::make_shared<RectangularMesh<2>>(cmesh, plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
        return interpolate(src_mesh, params, dest_mesh, interp,
                           InterpolationFlags(SOLVER->getGeometry(),
                                              symmetric()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                              InterpolationFlags::Symmetry::NO)
                          );
    }
}

void ExpansionPW2D::make_permeability_matrices(cmatrix& work)
{
    coeff_matrix_rmyy.reset(N, N);

    int order = int(SOLVER->getSize());

    if (symmetric()) {
        const bool sym = symmetry == E_LONG;
        for (int i = 0; i <= order; ++i)
            work(i,0) = muyy(i);
        for (int j = 1; j <= order; ++j)
            for (int i = 0; i <= order; ++i)
                work(i,j) = sym? muyy(abs(i-j)) - muyy(i+j) : muyy(abs(i-j)) + muyy(i+j);
        make_unit_matrix(coeff_matrix_rmyy);
        invmult(work, coeff_matrix_rmyy);

        if (polarization != E_TRAN) {
            coeff_matrix_mxx.reset(N, N);
            for (int i = 0; i <= order; ++i)
                work(i,0) = rmuxx(i);
            for (int j = 1; j <= order; ++j)
                for (int i = 0; i <= order; ++i)
                    work(i,j) = sym? rmuxx(abs(i-j)) + rmuxx(i+j) : rmuxx(abs(i-j)) - rmuxx(i+j);
            make_unit_matrix(coeff_matrix_mxx);
            invmult(work, coeff_matrix_mxx);
        }
    } else {
        for (int j = -order; j <= order; ++j) {
            const size_t jt = iEH(j);
            for (int i = -order; i <= order; ++i) {
                const size_t it = iEH(i);
                work(it,jt) = muyy(i-j);
            }
        }
        make_unit_matrix(coeff_matrix_rmyy);
        invmult(work, coeff_matrix_rmyy);

        if (polarization != E_TRAN) {
            coeff_matrix_mxx.reset(N, N);
            for (int j = -order; j <= order; ++j) {
                const size_t jt = iEH(j);
                for (int i = -order; i <= order; ++i) {
                    const size_t it = iEH(i);
                    work(it,jt) = rmuxx(i-j);
                }
            }
            make_unit_matrix(coeff_matrix_mxx);
            invmult(work, coeff_matrix_mxx);
        }
    }
}


void ExpansionPW2D::getMatrices(size_t l, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    if (coeff_matrices.empty()) coeff_matrices.resize(solver->lcount);

    dcomplex beta{ this->beta.real(),  this->beta.imag() - SOLVER->getMirrorLosses(this->beta.real()/k0.real()) };

    const int order = int(SOLVER->getSize());
    dcomplex rk0 = 1. / k0, k02 = k0*k0;
    double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);
    double bb = b * b;

    // Ez represents -Ez

    if (separated()) {
        if (symmetric()) {
            // Separated symmetric()
            if (polarization == E_LONG) {                   // Ez & Hx
                const bool sym = symmetry == E_LONG;
                if (!periodic) {
                    coeff_matrix_rmyy.copyto(RE);
                    coeff_matrix_mxx.copyto(RH);
                } else {
                    make_unit_matrix(RE);
                    make_unit_matrix(RH);
                }
                for (int i = 0; i <= order; ++i) {
                    dcomplex gi = - rk0 * bb * double(i);
                    RE(i,0) = k0 * epszz(l,i);
                    RH(i,0) *= k0;
                    for (int j = 1; j <= order; ++j) {
                        RE(i,j) = gi * double(j) * RE(i,j) + k0 * (sym? epszz(l,abs(i-j)) + epszz(l,i+j) : epszz(l,abs(i-j)) - epszz(l,i+j));
                        RH(i,j) *= k0;
                    }
                    // Ugly hack to avoid singularity
                    if (RE(i,i) == 0.) RE(i,i) = 1e-32;
                    if (RH(i,i) == 0.) RH(i,i) = 1e-32;
                }
            } else {                                        // Ex & Hz
                const bool sym = symmetry == E_TRAN;
                coeff_matrices[l].exx.copyto(RE);
                coeff_matrices[l].reyy.copyto(RH);
                for (int i = 0; i <= order; ++i) {
                    const dcomplex gi = - rk0 * bb * double(i);
                    RE(i,0) *= k0;
                    RH(i,0) = k0 * muzz(i);
                    for (int j = 1; j <= order; ++j) {
                        RE(i,j) *= k0;
                        RH(i,j) = gi * double(j) * RH(i,j) + k0 * (sym? muzz(abs(i-j)) + muzz(i+j) : muzz(abs(i-j)) - muzz(i+j));
                    }
                    // Ugly hack to avoid singularity
                    if (RE(i,i) == 0.) RE(i,i) = 1e-32;
                    if (RH(i,i) == 0.) RH(i,i) = 1e-32;
                }
            }
        } else {
            // Separated asymmetric()
            if (polarization == E_LONG) {                   // Ez & Hx
                if (!periodic) {
                    coeff_matrix_rmyy.copyto(RE);
                    coeff_matrix_mxx.copyto(RH);
                } else {
                    make_unit_matrix(RE);
                    make_unit_matrix(RH);
                }
                for (int i = -order; i <= order; ++i) {
                    const dcomplex gi = - rk0 * (b * double(i) - ktran);
                    const size_t it = iEH(i);
                    for (int j = -order; j <= order; ++j) {
                        const size_t jt = iEH(j);
                        RE(it,jt) = gi * (b * double(j) - ktran) * RE(it,jt) + k0 * epszz(l,i-j);
                        RH(it,jt) *= k0;
                    }
                    // Ugly hack to avoid singularity
                    if (RE(it,it) == 0.) RE(it,it) = 1e-32;
                    if (RH(it,it) == 0.) RH(it,it) = 1e-32;
                }
            } else {                                        // Ex & Hz
                coeff_matrices[l].exx.copyto(RE);
                coeff_matrices[l].reyy.copyto(RH);
                for (int i = -order; i <= order; ++i) {
                    const dcomplex gi = - rk0 * (b * double(i) - ktran);
                    const size_t it = iEH(i);
                    for (int j = -order; j <= order; ++j) {
                        const size_t jt = iEH(j);
                        RE(it,jt) *= k0;
                        RH(it,jt) = gi * (b * double(j) - ktran) * RH(it,jt) + k0 * muzz(i-j);
                    }
                    // Ugly hack to avoid singularity
                    if (RE(it,it) == 0.) RE(it,it) = 1e-32;
                    if (RH(it,it) == 0.) RH(it,it) = 1e-32;
                }
            }
        }
    } else {
        // work matrix is 2N×2N, so we can use its space for four N×N matrices
        const size_t NN = N*N;
        TempMatrix temp = getTempMatrix();
        cmatrix work(temp);
        cmatrix workxx, workyy;
        if (symmetric()) {
            // Full symmetric()
            const bool sel = symmetry == E_LONG;
            workyy = coeff_matrices[l].reyy;
            if (!periodic) {
                workxx = coeff_matrix_mxx;
            } else {
                workxx = cmatrix(N, N, work.data()+NN);
                make_unit_matrix(workxx);
            }
            for (int i = 0; i <= order; ++i) {
                const dcomplex gi = b * double(i) - ktran;
                const size_t iex = iEx(i), iez = iEz(i);
                for (int j = 0; j <= order; ++j) {
                    int ijp = abs(i-j), ijn = i+j;
                    dcomplex gj = b * double(j) - ktran;
                    const size_t jhx = iHx(j), jhz = iHz(j);
                    RH(iex,jhz) = - rk0 *  gi * gj  * workyy(i,j) +
                                    k0 * (j == 0? muzz(i) : sel? muzz(ijp) - muzz(ijn) : muzz(ijp) + muzz(ijn));
                    RH(iex,jhx) = - rk0 * beta* gi  * workyy(i,j);
                    RH(iez,jhz) = - rk0 * beta* gj  * workyy(i,j);
                    RH(iez,jhx) = - rk0 * beta*beta * workyy(i,j) + k0 * workxx(i,j);
                }
                // Ugly hack to avoid singularity
                if (RH(iex,iex) == 0.) RH(iex,iex) = 1e-32;
                if (RH(iez,iez) == 0.) RH(iez,iez) = 1e-32;
            }
            workxx = coeff_matrices[l].exx;
            if (!periodic) {
                workyy = coeff_matrix_rmyy;
            } else {
                workyy = cmatrix(N, N, work.data()+2*NN);
                make_unit_matrix(workyy);
            }
            for (int i = 0; i <= order; ++i) {
                const dcomplex gi = b * double(i) - ktran;
                const size_t ihx = iHx(i), ihz = iHz(i);
                for (int j = 0; j <= order; ++j) {
                    int ijp = abs(i-j), ijn = i+j;
                    dcomplex gj = b * double(j) - ktran;
                    const size_t jex = iEx(j), jez = iEz(j);
                    RE(ihz,jex) = - rk0 * beta*beta * workyy(i,j) + k0 * workxx(i,j);
                    RE(ihz,jez) =   rk0 * beta* gj  * workyy(i,j);
                    RE(ihx,jex) =   rk0 * beta* gi  * workyy(i,j);
                    RE(ihx,jez) = - rk0 *  gi * gj  * workyy(i,j) +
                                    k0 * (j == 0? epszz(l,i) : sel? epszz(l,ijp) + epszz(l,ijn) : epszz(l,ijp) - epszz(l,ijn));
                }
                // Ugly hack to avoid singularity
                if (RE(ihx,ihx) == 0.) RE(ihx,ihx) = 1e-32;
                if (RE(ihz,ihz) == 0.) RE(ihz,ihz) = 1e-32;
            }
        } else {
            // Full asymmetric()
            workyy = coeff_matrices[l].reyy;
            if (!periodic) {
                workxx = coeff_matrix_mxx;
            } else {
                workxx = cmatrix(N, N, work.data()+NN);
                make_unit_matrix(workxx);
            }
            for (int i = -order; i <= order; ++i) {
                const dcomplex gi = b * double(i) - ktran;
                const size_t iex = iEx(i), iez = iEz(i), it = iEH(i);
                for (int j = -order; j <= order; ++j) {
                    int ij = i-j;   dcomplex gj = b * double(j) - ktran;
                    const size_t jhx = iHx(j), jhz = iHz(j), jt = iEH(j);
                    RH(iex,jhz) = - rk0 *  gi * gj  * workyy(it,jt) + k0 * muzz(ij);
                    RH(iex,jhx) = - rk0 * beta* gi  * workyy(it,jt);
                    RH(iez,jhz) = - rk0 * beta* gj  * workyy(it,jt);
                    RH(iez,jhx) = - rk0 * beta*beta * workyy(it,jt) + k0 * workxx(it,jt);
                }
                // Ugly hack to avoid singularity
                if (RH(iex,iex) == 0.) RH(iex,iex) = 1e-32;
                if (RH(iez,iez) == 0.) RH(iez,iez) = 1e-32;
            }
            workxx = coeff_matrices[l].exx;
            if (!periodic) {
                workyy = coeff_matrix_rmyy;
            } else {
                workyy = cmatrix(N, N, work.data()+2*NN);
                make_unit_matrix(workyy);
            }
            for (int i = -order; i <= order; ++i) {
                const dcomplex gi = b * double(i) - ktran;
                const size_t ihx = iHx(i), ihz = iHz(i), it = iEH(i);
                for (int j = -order; j <= order; ++j) {
                    int ij = i-j;   dcomplex gj = b * double(j) - ktran;
                    const size_t jex = iEx(j), jez = iEz(j), jt = iEH(j);
                    RE(ihz,jex) = - rk0 * beta*beta * workyy(it,jt) + k0 * workxx(it,jt);
                    RE(ihz,jez) =   rk0 * beta* gj  * workyy(it,jt);
                    RE(ihx,jex) =   rk0 * beta* gi  * workyy(it,jt);
                    RE(ihx,jez) = - rk0 *  gi * gj  * workyy(it,jt) + k0 * epszz(l,ij);
                    if (epszx(l)) {
                        RE(ihx,jex) -= k0 * epszx(l,ij);
                        RE(ihz,jez) -= k0 * epsxz(l,ij);
                    }
                }
                // Ugly hack to avoid singularity
                if (RE(ihx,ihx) == 0.) RE(ihx,ihx) = 1e-32;
                if (RE(ihz,ihz) == 0.) RE(ihz,ihz) = 1e-32;
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

    TempMatrix temp = getTempMatrix();
    cvector work(temp.data(), N);

    if (which_field == FIELD_E) {
        if (polarization == E_LONG) {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                size_t ieh = iEH(i);
                field[ieh].tran() = field[ieh].vert() = 0.;
                if (ieh != 0 || !dz) field[ieh-dz].lon() = - E[ieh];
            }
        } else if (polarization == E_TRAN) {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                size_t ieh = iEH(i);
                field[ieh].lon() = 0.;
                if (ieh != 0 || !dx)
                    field[ieh-dx].tran() = E[ieh];
                if (ieh != 0 || !dz) {
                    field[ieh-dz].vert() = 0.;
                    for (int j = symmetric()? 0 : -order; j <= order; ++j) {
                        size_t jeh = iEH(j);
                        field[ieh-dz].vert() += coeff_matrices[l].reyy(ieh,jeh) * (b*double(j)-ktran) * H[jeh];
                    }
                    field[ieh-dz].vert() /= k0;
                }
            }
        } else {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                size_t ieh = iEH(i);
                if (ieh != 0 || !dx)
                    field[ieh-dx].tran() = E[iEx(i)];
                if (ieh != 0 || !dz) {
                    field[ieh-dz].lon() = - E[iEz(i)];
                    field[ieh-dz].vert() = 0.;
                    for (int j = symmetric()? 0 : -order; j <= order; ++j) {
                        field[ieh-dz].vert() += coeff_matrices[l].reyy(ieh,iEH(j))
                                                * ((b*double(j)-ktran) * H[iHz(j)] - beta * H[iHx(j)]);
                    }
                    field[ieh-dz].vert() /= k0;
                }
            }
        }
    } else { // which_field == FIELD_H
        if (polarization == E_TRAN) {  // polarization == H_LONG
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                size_t ieh = iEH(i);
                field[ieh].tran() = field[ieh].vert() = 0.;
                if (ieh != 0 || !dz) field[ieh-dz].lon() = H[ieh];
            }
        } else if (polarization == E_LONG) {  // polarization == H_TRAN
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                size_t ieh = iEH(i);
                field[ieh].lon() = 0.;
                if (ieh != 0 || !dx)
                    field[ieh-dx].tran() = H[ieh];
                if (ieh != 0 || !dz) {
                    if (periodic)
                        field[ieh-dz].vert() = - (b*double(i)-ktran) * E[ieh];
                    else {
                        field[ieh-dz].vert() = 0.;
                        for (int j = symmetric()? 0 : -order; j <= order; ++j) {
                            size_t jeh = iEH(j);
                            field[ieh-dz].vert() -= coeff_matrix_rmyy(ieh,jeh) * (b*double(j)-ktran) * E[jeh];
                        }
                    }
                    field[ieh-dz].vert() /= k0;
                }
            }
        } else {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                size_t ieh = iEH(i);
                if (ieh != 0 || !dx)
                    field[ieh-dx].tran() = H[iHx(i)];
                if (ieh != 0 || !dz) {
                    field[ieh-dz].lon() = H[iHz(i)];
                    if (periodic)
                        field[ieh-dz].vert() = (beta * E[iEx(i)] - (b*double(i)-ktran) * E[iEz(i)]);
                    else {
                        field[ieh-dz].vert() = 0.;
                        for (int j = symmetric()? 0 : -order; j <= order; ++j)
                            field[ieh-dz].vert() += coeff_matrix_rmyy(ieh,iEH(j))
                                                    * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
                    }
                    field[ieh-dz].vert() /= k0;
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
                    dcomplex sn =  2. * I * sin(B * k * x);
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
            if (sym == E_TRAN) {
                for (Vec<3,dcomplex>& f: field) {
                    f.c0 = dcomplex(-f.c0.imag(), f.c0.real());
                    f.c2 = dcomplex(-f.c2.imag(), f.c2.real());

                }
            } else {
                for (Vec<3,dcomplex>& f: field) {
                    f.c1 = dcomplex(-f.c1.imag(), f.c1.real());

                }
            }
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
            LazyData<Vec<3,dcomplex>> interpolated =
                interpolate(src_mesh, field, dest_mesh, field_interpolation,
                            InterpolationFlags(SOLVER->getGeometry(),
                            InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::NO),
                            false);
            dcomplex ikx = I * ktran;
            return LazyData<Vec<3,dcomplex>>(interpolated.size(), [interpolated, dest_mesh, ikx] (size_t i) {
                return interpolated[i] * exp(-ikx * dest_mesh->at(i).c0);
            });
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
                P += real(E[iEH(i)] * conj(H[iEH(i)]));
            }
            P = 2. * P - real(E[iEH(0)] * conj(H[iEH(0)]));
        } else {
            for (int i = -ord; i <= ord; ++i) {
                P += real(E[iEH(i)] * conj(H[iEH(i)]));
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

    return P * (symmetric()? 2 * right : right - left) * 1e-6; // µm² -> m²
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

double ExpansionPW2D::integrateField(WhichField field, size_t l, const cvector& E, const cvector& H)
{
    const int order = int(SOLVER->getSize());
    double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);

    dcomplex vert;
    double sum = 0.;

    if (which_field == FIELD_E) {
        if (polarization == E_TRAN) {
            if (symmetric()) {
                for (int i = 0; i <= order; ++i) {
                    vert = 0.;
                    for (int j = 0; j <= order; ++j)
                        vert += coeff_matrices[l].reyy(i,j) * b*double(j) * H[j];
                    vert /= k0;
                    sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
                }
            } else {
                for (int i = -order; i <= order; ++i) {
                    vert = 0.; // beta is equal to 0
                    size_t ieh = iEH(i);
                    for (int j = -order; j <= order; ++j) {
                        size_t jeh = iEH(j);
                        vert += coeff_matrices[l].reyy(ieh,jeh) * (b*double(j)-ktran) * H[jeh];
                    }
                    vert /= k0;
                    sum += real(vert * conj(vert));
                }
            }
        } else if (polarization != E_LONG) {
            if (symmetric()) {
                for (int i = 0; i <= order; ++i) {
                    vert = 0.;
                    for (int j = 0; j <= order; ++j)
                        vert -= coeff_matrices[l].reyy(i,j) * (beta * H[iHx(j)] + (b*double(j)-ktran) * H[iHz(j)]);
                    vert /= k0;
                    sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
                }
            } else {
                for (int i = -order; i <= order; ++i) {
                    vert = 0.;
                    for (int j = -order; j <= order; ++j)
                        vert -= coeff_matrices[l].reyy(iEH(i),iEH(j)) * (beta * H[iHx(j)] + (b*double(j)-ktran) * H[iHz(j)]);
                    vert /= k0;
                    sum += real(vert * conj(vert));
                }
            }
        }
    } else { // which_field == FIELD_H
        if (polarization == E_LONG) {  // polarization == H_TRAN
            if (symmetric()) {
                for (int i = 0; i <= order; ++i) {
                    if (periodic)
                        vert = - (b*double(i)-ktran) * E[iEH(i)];
                    else {
                        vert = 0.;
                        for (int j = 0; j <= order; ++j)
                            vert -= coeff_matrix_rmyy(i,j) * (b*double(j)-ktran) * E[iEH(j)];
                    }
                    vert /= k0;
                    sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
                }
            } else {
                for (int i = -order; i <= order; ++i) {
                    if (periodic)
                        vert = - (b*double(i)-ktran) * E[iEH(i)];
                    else {
                        vert = 0.;
                        for (int j = -order; j <= order; ++j)
                            vert -= coeff_matrix_rmyy(i,j) * (b*double(j)-ktran) * E[iEH(j)];
                    }
                    vert /= k0;
                    sum += real(vert * conj(vert));
                }
            }
        } else if (polarization != E_TRAN) {
            for (int i = symmetric()? 0 : -order; i <= order; ++i) {
                if (symmetric()) {
                    if (periodic)
                        vert = (beta * E[iEx(i)] - (b*double(i)-ktran) * E[iEz(i)]);
                    else {
                        vert = 0.;
                        for (int j = 0; j <= order; ++j)
                            vert += coeff_matrix_rmyy(i,j) * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
                    }
                    vert /= k0;
                    sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
                } else {
                    if (periodic)
                        vert = (beta * E[iEx(i)] - (b*double(i)-ktran) * E[iEz(i)]);
                    else {
                        vert = 0.;
                        for (int j = -order; j <= order; ++j)
                            vert += coeff_matrix_rmyy(i,j) * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
                    }
                    vert /= k0;
                    sum += real(vert * conj(vert));
                }
            }
        }
    }

    double L = (right-left) * (symmetric()? 2. : 1.);
    if (field == FIELD_E) {
        if (symmetric()) {
            for (dcomplex e: E) sum += 2. * real(e * conj(e));
            if (separated()) {
                size_t i = iEH(0);
                sum -= real(E[i] * conj(E[i]));
            } else {
                size_t ix = iEx(0), iz = iEz(0);
                sum -= real(E[ix] * conj(E[ix])) + real(E[iz] * conj(E[iz]));
            }
        } else {
            for (dcomplex e: E) sum += real(e * conj(e));
        }
    } else {
        if (symmetric()) {
            for (dcomplex h: H) sum += 2. * real(h * conj(h));
            if (separated()) {
                size_t i = iEH(0);
                sum -= real(H[i] * conj(H[i]));
            } else {
                size_t ix = iHx(0), iz = iHz(0);
                sum -= real(H[ix] * conj(H[ix])) + real(H[iz] * conj(H[iz]));
            }
        } else {
            for (dcomplex e: H) sum += real(e * conj(e));
        }
    }
    return 0.5 * L * sum;
}


}}} // namespace plask
