#include <boost/algorithm/clamp.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/adaptor/transformed.hpp>
using boost::algorithm::clamp;

#include "../meshadapter.hpp"
#include "expansion2d.hpp"
#include "solver2d.hpp"

#define SOLVER static_cast<LinesSolver2D*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionFD2D::ExpansionFD2D(LinesSolver2D* solver)
    : Expansion(solver), initialized(false), symmetry(E_UNSPECIFIED), polarization(E_UNSPECIFIED) {}

void ExpansionFD2D::setPolarization(Component pol) {
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

void ExpansionFD2D::init() {
    auto geometry = SOLVER->getGeometry();

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    auto bbox = geometry->getChild()->getBoundingBox();
    left = bbox.lower[0];
    right = bbox.upper[0];

    size_t refine = SOLVER->refine;
    if (refine == 0) refine = 1;

    if (symmetry != E_UNSPECIFIED && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
        throw BadInput(solver->getId(), "Symmetry not allowed for asymmetric structure");

    if (geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN)) {
        if (right <= 0.) {
            left = -left;
            right = -right;
            std::swap(left, right);
        }
        if (left != 0.) throw BadMesh(SOLVER->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric()) left = -right;
    }

    double dx;
    size_t N;

    if (SOLVER->mesh) {
        if (SOLVER->mesh->size() < 2) throw BadInput(SOLVER->getId(), "Mesh needs at least two points");
        if (!is_zero(SOLVER->mesh->at(0) - (symmetric() ? -right : left)))
            throw BadInput(SOLVER->getId(), "First mesh point ({}) must match left geometry boundary ({})", SOLVER->mesh->at(0),
                           symmetric() ? -right : left);
        if (!is_zero(SOLVER->mesh->at(SOLVER->mesh->size() - 1) - right))
            throw BadInput(SOLVER->getId(), "Last mesh point ({}) must match right geometry boundary ({})",
                           SOLVER->mesh->at(SOLVER->mesh->size() - 1), right);
        dx = SOLVER->mesh->step();
        N = SOLVER->mesh->size();
    } else {
        if (isnan(SOLVER->density)) throw Exception(format("{0}: No mesh nor density specified", SOLVER->getId()));
        double L = right - left;
        N = size_t(std::ceil(L / SOLVER->density)) + 1;
        dx = L / (N - 1);
    }

    if (periodic && !symmetric()) {
        right -= dx;
        N -= 1;
    }

    double xs = dx * (0.5 - 0.5 / refine);
    auto xmesh = plask::make_shared<RegularAxis>(left - xs, right + xs, N * refine);

    // Add PMLs
    if (!periodic) {
        pml = SOLVER->pml;
        double nd = std::ceil(pml.dist / dx), np = std::ceil(pml.size / dx);
        pml.dist = nd * dx;
        pml.size = np * dx;
        right += pml.size + pml.dist;
        ndr = size_t(nd);
        npr = size_t(np);
        if (!symmetric()) {
            left -= pml.size + pml.dist;
            ndl = ndr;
            npl = npr;
        } else {
            ndl = 0;
            npl = 0;
        }
        N += ndl + npl + ndr + npr;
    } else {
        npl = ndl = ndr = npr = 0;
    }

    mesh = plask::make_shared<RegularAxis>(left, right, N);

    size_t M = matrixSize();
    SOLVER->writelog(LOG_DETAIL, "Creating{2}{3} discretization with {0} points (matrix size: {1})", N, M,
                     symmetric() ? " symmetric" : "", separated() ? " separated" : "");

    if (symmetric()) SOLVER->writelog(LOG_DETAIL, "Symmetry is {0}", (symmetry == E_TRAN) ? "Etran" : "Elong");

    // Compute permeability coefficients
    mag.reset(mesh->size(), 1.);
    if (!periodic) {
        for (size_t i = 0; i < npl; ++i) {
            double h = (npl - i) * dx;
            mag[i] = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
        }
        for (size_t i = 0, n = N - npr; i < npr; ++i) {
            double h = i * dx;
            mag[n + i] = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
        }
    }

    material_mesh = plask::make_shared<RectangularMesh<2>>(xmesh, solver->verts, RectangularMesh<2>::ORDER_01);

    initialized = true;

    // Allocate memory for expansion coefficients
    size_t nlayers = solver->lcount;
    diagonals.assign(nlayers, false);
    epsilon.resize(solver->lcount);
}

void ExpansionFD2D::reset() {
    epsilon.clear();
    initialized = false;
    mesh.reset();
    mag.reset();
    temporary.reset();
}

void ExpansionFD2D::beforeLayersIntegrals(double lam, double glam) {
    SOLVER->prepareExpansionIntegrals(this, material_mesh, lam, glam);
}

void ExpansionFD2D::layerIntegrals(size_t layer, double lam, double glam) {
    auto geometry = SOLVER->getGeometry();

    size_t refine = SOLVER->refine;
    if (refine == 0) refine = 1;

    double maty;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            maty = solver->verts->at(i);
            break;
        }
    }

    diagonals[layer] = true;

    size_t shift = npl + ndl;
    size_t N = mesh->size() - shift - npr - ndr;

#if defined(OPENMP_FOUND)  // && !defined(NDEBUG)
    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points) in thread {}", layer,
                     solver->lcount, refine * N, omp_get_thread_num());
#else
    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points)", layer, solver->lcount,
                     refine * N);
#endif

    if (isnan(lam)) throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    epsilon[layer].reset(mesh->size(), Tensor3<dcomplex>(0.));

    for (size_t i = 0; i < N; i++) {
        size_t i1 = i + shift;
        size_t j0 = i * refine;
        for (size_t j = 0; j < refine; ++j) {
            Tensor3<dcomplex> eps = getEpsilon(geometry, layer, maty, lam, glam, j0 + j);
            if (eps.c01 != 0. && separated())
                throw BadInput(solver->getId(), "Polarization can be specified only for diagonal refractive index tensor (NR)");
            if (eps.c01 != 0. && symmetric())
                throw BadInput(solver->getId(), "Symmetry can be specified only for diagonal refractive index tensor (NR)");
            epsilon[layer][i1] += eps;
        }
        epsilon[layer][i1] /= refine;
    }

    // Add PMLs
    if (!periodic) {
        double dx = mesh->at(1) - mesh->at(0);
        auto epsl = epsilon[layer][shift];
        std::fill(epsilon[layer].begin() + npl, epsilon[layer].begin() + shift, epsl);
        for (size_t i = 0; i < npl; ++i) {
            double h = (npl - i) * dx;
            dcomplex s = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
            epsilon[layer][i] = Tensor3<dcomplex>(epsl.c00 * s, epsl.c11 / s, epsl.c22 * s);
        }
        size_t idr = mesh->size() - npr - ndr - 1;
        auto epsr = epsilon[layer][idr];
        std::fill(epsilon[layer].begin() + idr, epsilon[layer].begin() + idr + ndr, epsr);
        for (size_t i = 0, ipr = idr + ndr; i <= npr; ++i) {
            double h = i * dx;
            dcomplex s = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
            epsilon[layer][ipr + i] = Tensor3<dcomplex>(epsr.c00 * s, epsr.c11 / s, epsr.c22 * s);
        }
    }

    Tensor3<dcomplex> eps0 = epsilon[layer][0];
    for (Tensor3<dcomplex>& eps : epsilon[layer]) {
        if (!is_zero(eps - eps0)) diagonals[layer] = false;
        eps.c22 = 1. / eps.c22;
    }

    if (diagonals[layer]) SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
}

LazyData<Tensor3<dcomplex>> ExpansionFD2D::getMaterialNR(size_t l,
                                                         const shared_ptr<const LevelsAdapter::Level>& level,
                                                         InterpolationMethod interp) {
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(mesh, plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));

    DataVector<Tensor3<dcomplex>> nr(this->epsilon[l].size());
    Tensor3<dcomplex>* to = nr.begin();
    for (const Tensor3<dcomplex>& val: this->epsilon[l]) {
        *(to++) = Tensor3<dcomplex>(sqrt(val.c00), sqrt(val.c11), sqrt(1. / val.c22), sqrt(val.c01));
    }

    return interpolate(src_mesh, nr, dest_mesh, getInterpolationMethod<INTERPOLATION_LINEAR>(interp),
                       InterpolationFlags(SOLVER->getGeometry(),
                                          symmetric() ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                          InterpolationFlags::Symmetry::NO));
}

void ExpansionFD2D::getMatrices(size_t l, cmatrix& RE, cmatrix& RH) {
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    // if (coeff_matrices.empty()) coeff_matrices.resize(solver->lcount);

    // dcomplex beta{ this->beta.real(),  this->beta.imag() - SOLVER->getMirrorLosses(this->beta.real()/k0.real()) };

    // const int order = int(SOLVER->getSize());
    // dcomplex rk0 = 1. / k0, k02 = k0*k0;
    // double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);
    // double bb = b * b;

    // // Ez represents -Ez

    // if (separated()) {
    //     if (symmetric()) {
    //         // Separated symmetric()
    //         if (polarization == E_LONG) {                   // Ez & Hx
    //             const bool sym = symmetry == E_LONG;
    //             if (!periodic) {
    //                 coeff_matrix_rmyy.copyto(RE);
    //                 coeff_matrix_mxx.copyto(RH);
    //             } else {
    //                 make_unit_matrix(RE);
    //                 make_unit_matrix(RH);
    //             }
    //             for (int i = 0; i <= order; ++i) {
    //                 dcomplex gi = - rk0 * bb * double(i);
    //                 RE(i,0) = k0 * epszz(l,i);
    //                 RH(i,0) *= k0;
    //                 for (int j = 1; j <= order; ++j) {
    //                     RE(i,j) = gi * double(j) * RE(i,j) + k0 * (sym? epszz(l,abs(i-j)) + epszz(l,i+j) : epszz(l,abs(i-j)) -
    //                     epszz(l,i+j)); RH(i,j) *= k0;
    //                 }
    //                 // Ugly hack to avoid singularity
    //                 if (RE(i,i) == 0.) RE(i,i) = 1e-32;
    //                 if (RH(i,i) == 0.) RH(i,i) = 1e-32;
    //             }
    //         } else {                                        // Ex & Hz
    //             const bool sym = symmetry == E_TRAN;
    //             coeff_matrices[l].exx.copyto(RE);
    //             coeff_matrices[l].reyy.copyto(RH);
    //             for (int i = 0; i <= order; ++i) {
    //                 const dcomplex gi = - rk0 * bb * double(i);
    //                 RE(i,0) *= k0;
    //                 RH(i,0) = k0 * muzz(i);
    //                 for (int j = 1; j <= order; ++j) {
    //                     RE(i,j) *= k0;
    //                     RH(i,j) = gi * double(j) * RH(i,j) + k0 * (sym? muzz(abs(i-j)) + muzz(i+j) : muzz(abs(i-j)) - muzz(i+j));
    //                 }
    //                 // Ugly hack to avoid singularity
    //                 if (RE(i,i) == 0.) RE(i,i) = 1e-32;
    //                 if (RH(i,i) == 0.) RH(i,i) = 1e-32;
    //             }
    //         }
    //     } else {
    //         // Separated asymmetric()
    //         if (polarization == E_LONG) {                   // Ez & Hx
    //             if (!periodic) {
    //                 coeff_matrix_rmyy.copyto(RE);
    //                 coeff_matrix_mxx.copyto(RH);
    //             } else {
    //                 make_unit_matrix(RE);
    //                 make_unit_matrix(RH);
    //             }
    //             for (int i = -order; i <= order; ++i) {
    //                 const dcomplex gi = - rk0 * (b * double(i) - ktran);
    //                 const size_t it = iEH(i);
    //                 for (int j = -order; j <= order; ++j) {
    //                     const size_t jt = iEH(j);
    //                     RE(it,jt) = gi * (b * double(j) - ktran) * RE(it,jt) + k0 * epszz(l,i-j);
    //                     RH(it,jt) *= k0;
    //                 }
    //                 // Ugly hack to avoid singularity
    //                 if (RE(it,it) == 0.) RE(it,it) = 1e-32;
    //                 if (RH(it,it) == 0.) RH(it,it) = 1e-32;
    //             }
    //         } else {                                        // Ex & Hz
    //             coeff_matrices[l].exx.copyto(RE);
    //             coeff_matrices[l].reyy.copyto(RH);
    //             for (int i = -order; i <= order; ++i) {
    //                 const dcomplex gi = - rk0 * (b * double(i) - ktran);
    //                 const size_t it = iEH(i);
    //                 for (int j = -order; j <= order; ++j) {
    //                     const size_t jt = iEH(j);
    //                     RE(it,jt) *= k0;
    //                     RH(it,jt) = gi * (b * double(j) - ktran) * RH(it,jt) + k0 * muzz(i-j);
    //                 }
    //                 // Ugly hack to avoid singularity
    //                 if (RE(it,it) == 0.) RE(it,it) = 1e-32;
    //                 if (RH(it,it) == 0.) RH(it,it) = 1e-32;
    //             }
    //         }
    //     }
    // } else {
    //     // work matrix is 2N×2N, so we can use its space for four N×N matrices
    //     const size_t NN = N*N;
    //     TempMatrix temp = getTempMatrix();
    //     cmatrix work(temp);
    //     cmatrix workij(N, N, work.data());
    //     cmatrix workxx, workyy;
    //     if (symmetric()) {
    //         // Full symmetric()
    //         const bool sel = symmetry == E_LONG;
    //         workyy = coeff_matrices[l].reyy;
    //         if (!periodic) {
    //             workxx = coeff_matrix_mxx;
    //         } else {
    //             workxx = cmatrix(N, N, work.data()+NN);
    //             make_unit_matrix(workxx);
    //         }
    //         for (int i = 0; i <= order; ++i) {
    //             const dcomplex gi = b * double(i) - ktran;
    //             const size_t iex = iEx(i), iez = iEz(i);
    //             for (int j = 0; j <= order; ++j) {
    //                 int ijp = abs(i-j), ijn = i+j;
    //                 dcomplex gj = b * double(j) - ktran;
    //                 const size_t jhx = iHx(j), jhz = iHz(j);
    //                 dcomplex reyy = j == 0? repsyy(l,i) :
    //                                 sel? repsyy(l,ijp) + repsyy(l,ijn) : repsyy(l,ijp) - repsyy(l,ijn);
    //                 RH(iex,jhz) = - rk0 *  gi * gj  * workyy(i,j) +
    //                                 k0 * (j == 0? muzz(i) : sel? muzz(ijp) - muzz(ijn) : muzz(ijp) + muzz(ijn));
    //                 RH(iex,jhx) = - rk0 * beta* gi  * reyy;
    //                 RH(iez,jhz) = - rk0 * beta* gj  * workyy(i,j);
    //                 RH(iez,jhx) = - rk0 * beta*beta * reyy + k0 * workxx(i,j);
    //             }
    //             // Ugly hack to avoid singularity
    //             if (RH(iex,iex) == 0.) RH(iex,iex) = 1e-32;
    //             if (RH(iez,iez) == 0.) RH(iez,iez) = 1e-32;
    //         }
    //         workxx = coeff_matrices[l].exx;
    //         if (!periodic) {
    //             workyy = coeff_matrix_rmyy;
    //         } else {
    //             workyy = cmatrix(N, N, work.data()+2*NN);
    //             make_unit_matrix(workyy);
    //         }
    //         for (int i = 0; i <= order; ++i) {
    //             const dcomplex gi = b * double(i) - ktran;
    //             const size_t ihx = iHx(i), ihz = iHz(i);
    //             for (int j = 0; j <= order; ++j) {
    //                 int ijp = abs(i-j), ijn = i+j;
    //                 dcomplex gj = b * double(j) - ktran;
    //                 const size_t jex = iEx(j), jez = iEz(j);
    //                 dcomplex rmyy = j == 0? rmuyy(i) :
    //                                 sel? rmuyy(ijp) - rmuyy(ijn) : rmuyy(ijp) + rmuyy(ijn);
    //                 RE(ihz,jex) = - rk0 * beta*beta * rmyy + k0 * workxx(i,j);
    //                 RE(ihz,jez) =   rk0 * beta* gj  * workyy(i,j);
    //                 RE(ihx,jex) =   rk0 * beta* gi  * rmyy;
    //                 RE(ihx,jez) = - rk0 *  gi * gj  * workyy(i,j) +
    //                                 k0 * (j == 0? epszz(l,i) : sel? epszz(l,ijp) + epszz(l,ijn) : epszz(l,ijp) - epszz(l,ijn));
    //             }
    //             // Ugly hack to avoid singularity
    //             if (RE(ihx,ihx) == 0.) RE(ihx,ihx) = 1e-32;
    //             if (RE(ihz,ihz) == 0.) RE(ihz,ihz) = 1e-32;
    //         }
    //     } else {
    //         // Full asymmetric()
    //         workyy = coeff_matrices[l].reyy;
    //         if (!periodic) {
    //             workxx = coeff_matrix_mxx;
    //         } else {
    //             workxx = cmatrix(N, N, work.data()+NN);
    //             make_unit_matrix(workxx);
    //         }
    //         for (int i = -order; i <= order; ++i) {
    //             const dcomplex gi = b * double(i) - ktran;
    //             const size_t iex = iEx(i), iez = iEz(i), it = iEH(i);
    //             for (int j = -order; j <= order; ++j) {
    //                 int ij = i-j;   dcomplex gj = b * double(j) - ktran;
    //                 const size_t jhx = iHx(j), jhz = iHz(j), jt = iEH(j);
    //                 RH(iex,jhz) = - rk0 *  gi * gj  * workyy(it,jt) + k0 * muzz(ij);
    //                 RH(iex,jhx) = - rk0 * beta* gi  * repsyy(l,ij);
    //                 RH(iez,jhz) = - rk0 * beta* gj  * workyy(it,jt);
    //                 RH(iez,jhx) = - rk0 * beta*beta * repsyy(l,ij) + k0 * workxx(it,jt);
    //             }
    //             // Ugly hack to avoid singularity
    //             if (RH(iex,iex) == 0.) RH(iex,iex) = 1e-32;
    //             if (RH(iez,iez) == 0.) RH(iez,iez) = 1e-32;
    //         }
    //         workxx = coeff_matrices[l].exx;
    //         if (!periodic) {
    //             workyy = coeff_matrix_rmyy;
    //         } else {
    //             workyy = cmatrix(N, N, work.data()+2*NN);
    //             make_unit_matrix(workyy);
    //         }
    //         for (int i = -order; i <= order; ++i) {
    //             const dcomplex gi = b * double(i) - ktran;
    //             const size_t ihx = iHx(i), ihz = iHz(i), it = iEH(i);
    //             for (int j = -order; j <= order; ++j) {
    //                 int ij = i-j;   dcomplex gj = b * double(j) - ktran;
    //                 const size_t jex = iEx(j), jez = iEz(j), jt = iEH(j);
    //                 RE(ihz,jex) = - rk0 * beta*beta * rmuyy(ij) + k0 * workxx(it,jt);
    //                 RE(ihz,jez) =   rk0 * beta* gj  * workyy(it,jt);
    //                 RE(ihx,jex) =   rk0 * beta* gi  * rmuyy(ij);
    //                 RE(ihx,jez) = - rk0 *  gi * gj  * workyy(it,jt) + k0 * epszz(l,ij);
    //                 if (epszx(l)) {
    //                     RE(ihx,jex) -= k0 * coeff_matrices[l].ezx(it,jt);
    //                     RE(ihz,jez) -= k0 * epsxz(l,ij);
    //                 }
    //             }
    //             // Ugly hack to avoid singularity
    //             if (RE(ihx,ihx) == 0.) RE(ihx,ihx) = 1e-32;
    //             if (RE(ihz,ihz) == 0.) RE(ihz,ihz) = 1e-32;
    //         }
    //     }
    // }
}

void ExpansionFD2D::prepareField() {
    // if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_LINES;
    // if (symmetric()) {
    //     field.reset(N);
    //     if (field_interpolation != INTERPOLATION_LINES) {
    //         Component sym = (which_field == FIELD_E)? symmetry : Component(3-symmetry);
    //         int df = SOLVER->dct2()? 0 : 4;
    //         fft_x = FFT::Backward1D(1, N, FFT::Symmetry(sym+df), 3);    // tran
    //         fft_yz = FFT::Backward1D(1, N, FFT::Symmetry(3-sym+df), 3); // long
    //     }
    // } else {
    //     field.reset(N + 1);
    //     if (field_interpolation != INTERPOLATION_LINES)
    //         fft_x = FFT::Backward1D(3, N, FFT::SYMMETRY_NONE);
    // }
}

void ExpansionFD2D::cleanupField() { field.reset(); }

LazyData<Vec<3, dcomplex>> ExpansionFD2D::getField(size_t l,
                                                   const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                   const cvector& E,
                                                   const cvector& H) {
    // Component sym = (which_field == FIELD_E)? symmetry : Component((3-symmetry) % 3);

    // dcomplex beta{ this->beta.real(),  this->beta.imag() - SOLVER->getMirrorLosses(this->beta.real()/k0.real()) };

    // const int order = int(SOLVER->getSize());
    // double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);
    // assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    // auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    // double vpos = level->vpos();

    // int dx = (symmetric() && field_interpolation != INTERPOLATION_LINES && sym != E_TRAN)? 1 : 0; // 1 for sin expansion of tran
    // component int dz = (symmetric() && field_interpolation != INTERPOLATION_LINES && sym != E_LONG)? 1 : 0; // 1 for sin
    // expansion of long component

    // TempMatrix temp = getTempMatrix();
    // cvector work(temp.data(), N);

    // if (which_field == FIELD_E) {
    //     if (polarization == E_LONG) {
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             size_t ieh = iEH(i);
    //             field[ieh].tran() = field[ieh].vert() = 0.;
    //             if (ieh != 0 || !dz) field[ieh-dz].lon() = - E[ieh];
    //         }
    //     } else if (polarization == E_TRAN) {
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             size_t ieh = iEH(i);
    //             field[ieh].lon() = 0.;
    //             if (ieh != 0 || !dx)
    //                 field[ieh-dx].tran() = E[ieh];
    //             if (ieh != 0 || !dz) {
    //                 field[ieh-dz].vert() = 0.;
    //                 for (int j = symmetric()? 0 : -order; j <= order; ++j) {
    //                     size_t jeh = iEH(j);
    //                     field[ieh-dz].vert() += coeff_matrices[l].reyy(ieh,jeh) * (b*double(j)-ktran) * H[jeh];
    //                 }
    //                 field[ieh-dz].vert() /= k0;
    //             }
    //         }
    //     } else {
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             size_t ieh = iEH(i);
    //             if (ieh != 0 || !dx)
    //                 field[ieh-dx].tran() = E[iEx(i)];
    //             if (ieh != 0 || !dz) {
    //                 field[ieh-dz].lon() = - E[iEz(i)];
    //                 field[ieh-dz].vert() = 0.;
    //                 for (int j = symmetric()? 0 : -order; j <= order; ++j) {
    //                     field[ieh-dz].vert() += coeff_matrices[l].reyy(ieh,iEH(j))
    //                                             * ((b*double(j)-ktran) * H[iHz(j)] - beta * H[iHx(j)]);
    //                 }
    //                 field[ieh-dz].vert() /= k0;
    //             }
    //         }
    //     }
    // } else { // which_field == FIELD_H
    //     if (polarization == E_TRAN) {  // polarization == H_LONG
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             size_t ieh = iEH(i);
    //             field[ieh].tran() = field[ieh].vert() = 0.;
    //             if (ieh != 0 || !dz) field[ieh-dz].lon() = H[ieh];
    //         }
    //     } else if (polarization == E_LONG) {  // polarization == H_TRAN
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             size_t ieh = iEH(i);
    //             field[ieh].lon() = 0.;
    //             if (ieh != 0 || !dx)
    //                 field[ieh-dx].tran() = H[ieh];
    //             if (ieh != 0 || !dz) {
    //                 if (periodic)
    //                     field[ieh-dz].vert() = - (b*double(i)-ktran) * E[ieh];
    //                 else {
    //                     field[ieh-dz].vert() = 0.;
    //                     for (int j = symmetric()? 0 : -order; j <= order; ++j) {
    //                         size_t jeh = iEH(j);
    //                         field[ieh-dz].vert() -= coeff_matrix_rmyy(ieh,jeh) * (b*double(j)-ktran) * E[jeh];
    //                     }
    //                 }
    //                 field[ieh-dz].vert() /= k0;
    //             }
    //         }
    //     } else {
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             size_t ieh = iEH(i);
    //             if (ieh != 0 || !dx)
    //                 field[ieh-dx].tran() = H[iHx(i)];
    //             if (ieh != 0 || !dz) {
    //                 field[ieh-dz].lon() = H[iHz(i)];
    //                 if (periodic)
    //                     field[ieh-dz].vert() = (beta * E[iEx(i)] - (b*double(i)-ktran) * E[iEz(i)]);
    //                 else {
    //                     field[ieh-dz].vert() = 0.;
    //                     for (int j = symmetric()? 0 : -order; j <= order; ++j)
    //                         field[ieh-dz].vert() += coeff_matrix_rmyy(ieh,iEH(j))
    //                                                 * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
    //                 }
    //                 field[ieh-dz].vert() /= k0;
    //             }
    //         }
    //     }
    // }

    // if (dx) { field[field.size()-1].tran() = 0.; }
    // if (dz) { field[field.size()-1].lon() = 0.; field[field.size()-1].vert() = 0.; }

    // if (field_interpolation == INTERPOLATION_LINES) {
    //     DataVector<Vec<3,dcomplex>> result(dest_mesh->size());
    //     double L = right - left;
    //     if (!symmetric()) {
    //         dcomplex B = 2*PI * I / L;
    //         dcomplex ikx = I * ktran;
    //         result.reset(dest_mesh->size(), Vec<3,dcomplex>(0.,0.,0.));
    //         for (int k = -order; k <= order; ++k) {
    //             size_t j = (k>=0)? k : k + N;
    //             dcomplex G = B * double(k) - ikx;
    //             for (size_t i = 0; i != dest_mesh->size(); ++i) {
    //                 double x = dest_mesh->at(i)[0];
    //                 if (!periodic) x = clamp(x, left, right);
    //                 result[i] += field[j] * exp(G * (x-left));
    //             }
    //         }
    //     } else {
    //         double B = PI / L;
    //         result.reset(dest_mesh->size());
    //         for (size_t i = 0; i != dest_mesh->size(); ++i) {
    //             result[i] = field[0];
    //             for (int k = 1; k <= order; ++k) {
    //                 double x = dest_mesh->at(i)[0];
    //                 if (!periodic) x = clamp(x, -right, right);
    //                 double cs =  2. * cos(B * k * x);
    //                 dcomplex sn =  2. * I * sin(B * k * x);
    //                 if (sym == E_TRAN) {
    //                     result[i].lon() += field[k].lon() * sn;
    //                     result[i].tran() += field[k].tran() * cs;
    //                     result[i].vert() += field[k].vert() * sn;
    //                 } else {
    //                     result[i].lon() += field[k].lon() * cs;
    //                     result[i].tran() += field[k].tran() * sn;
    //                     result[i].vert() += field[k].vert() * cs;
    //                 }
    //             }
    //         }
    //     }
    //     return result;
    // } else {
    //     if (symmetric()) {
    //         fft_x.execute(&(field.data()->tran()));
    //         fft_yz.execute(&(field.data()->lon()));
    //         fft_yz.execute(&(field.data()->vert()));
    //         if (sym == E_TRAN) {
    //             for (Vec<3,dcomplex>& f: field) {
    //                 f.c0 = dcomplex(-f.c0.imag(), f.c0.real());
    //                 f.c2 = dcomplex(-f.c2.imag(), f.c2.real());

    //             }
    //         } else {
    //             for (Vec<3,dcomplex>& f: field) {
    //                 f.c1 = dcomplex(-f.c1.imag(), f.c1.real());

    //             }
    //         }
    //         double dx = 0.5 * (right-left) / double(N);
    //         auto src_mesh = plask::make_shared<RectangularMesh<2>>(plask::make_shared<RegularAxis>(left+dx, right-dx,
    //         field.size()), plask::make_shared<RegularAxis>(vpos, vpos, 1)); return interpolate(src_mesh, field, dest_mesh,
    //         field_interpolation,
    //                            InterpolationFlags(SOLVER->getGeometry(),
    //                                 (sym == E_TRAN)? InterpolationFlags::Symmetry::NPN : InterpolationFlags::Symmetry::PNP,
    //                                 InterpolationFlags::Symmetry::NO),
    //                                 false);
    //     } else {
    //         fft_x.execute(reinterpret_cast<dcomplex*>(field.data()));
    //         field[N] = field[0];
    //         auto src_mesh = plask::make_shared<RectangularMesh<2>>(plask::make_shared<RegularAxis>(left, right, field.size()),
    //         plask::make_shared<RegularAxis>(vpos, vpos, 1)); auto result = interpolate(src_mesh, field, dest_mesh,
    //         field_interpolation,
    //                                   InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::NO,
    //                                   InterpolationFlags::Symmetry::NO), false).claim();
    //         dcomplex ikx = I * ktran;
    //         for (size_t i = 0; i != dest_mesh->size(); ++i)
    //             result[i] *= exp(- ikx * dest_mesh->at(i).c0);
    //         return result;
    //     }
    // }
}

double ExpansionFD2D::integratePoyntingVert(const cvector& E, const cvector& H) {
    //     double P = 0.;

    //     const int ord = int(SOLVER->getSize());

    //     if (separated()) {
    //         if (symmetric()) {
    //             for (int i = 0; i <= ord; ++i) {
    //                 P += real(E[iEH(i)] * conj(H[iEH(i)]));
    //             }
    //             P = 2. * P - real(E[iEH(0)] * conj(H[iEH(0)]));
    //         } else {
    //             for (int i = -ord; i <= ord; ++i) {
    //                 P += real(E[iEH(i)] * conj(H[iEH(i)]));
    //             }
    //         }
    //     } else {
    //         if (symmetric()) {
    //             for (int i = 0; i <= ord; ++i) {
    //                 P -= real(E[iEz(i)] * conj(H[iHx(i)]) + E[iEx(i)] * conj(H[iHz(i)]));
    //             }
    //             P = 2. * P + real(E[iEz(0)] * conj(H[iHx(0)]) + E[iEx(0)] * conj(H[iHz(0)]));
    //         } else {
    //             for (int i = -ord; i <= ord; ++i) {
    //                 P -= real(E[iEz(i)] * conj(H[iHx(i)]) + E[iEx(i)] * conj(H[iHz(i)]));
    //             }
    //         }
    //     }

    //     double L = SOLVER->geometry->getExtrusion()->getLength();
    //     if (!isinf(L))
    //         P *= L * 1e-6;

    //     return P * (symmetric()? 2 * right : right - left) * 1e-6; // µm² -> m²
}

void ExpansionFD2D::getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix& RE, const cdiagonal& gamma) {
    size_t nr = Te.rows(), nc = Te.cols();
    std::fill_n(Te.data(), nr * nc, 0.);
    std::fill_n(Te1.data(), nr * nc, 0.);

    if (separated()) {
        for (std::size_t i = 0; i < nc; i++) {
            Te(i, i) = Te1(i, i) = 1.;
        }
    } else {
        // Ensure that for the same gamma E*H [2x2] is diagonal
        assert(nc % 2 == 0);
        size_t n = nc / 2;
        for (std::size_t i = 0; i < n; i++) {
            // Compute Te1 = sqrt(RE)
            // https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
            // but after this normalize columns to 1
            dcomplex a = RE(2 * i, 2 * i), b = RE(2 * i, 2 * i + 1), c = RE(2 * i + 1, 2 * i), d = RE(2 * i + 1, 2 * i + 1);
            dcomplex s = sqrt(a * d - b * c);
            a += s;
            d += s;
            // Normalize
            s = 1. / sqrt(a * a + b * b);
            a *= s;
            b *= s;
            s = 1. / sqrt(c * c + d * d);
            c *= s;
            d *= s;
            Te1(2 * i, 2 * i) = a;
            Te1(2 * i, 2 * i + 1) = b;
            Te1(2 * i + 1, 2 * i) = c;
            Te1(2 * i + 1, 2 * i + 1) = d;
            // Invert Te1
            s = 1. / (a * d - b * c);
            Te(2 * i, 2 * i) = s * d;
            Te(2 * i, 2 * i + 1) = -s * b;
            Te(2 * i + 1, 2 * i) = -s * c;
            Te(2 * i + 1, 2 * i + 1) = s * a;
        }
    }
}

double ExpansionFD2D::integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) {
    // const int order = int(SOLVER->getSize());
    // double b = 2.*PI / (right-left) * (symmetric()? 0.5 : 1.0);

    // dcomplex vert;
    // double sum = 0.;

    // if (which_field == FIELD_E) {
    //     if (polarization == E_TRAN) {
    //         if (symmetric()) {
    //             for (int i = 0; i <= order; ++i) {
    //                 vert = 0.;
    //                 for (int j = 0; j <= order; ++j)
    //                     vert += coeff_matrices[l].reyy(i,j) * b*double(j) * H[j];
    //                 vert /= k0;
    //                 sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
    //             }
    //         } else {
    //             for (int i = -order; i <= order; ++i) {
    //                 vert = 0.; // beta is equal to 0
    //                 size_t ieh = iEH(i);
    //                 for (int j = -order; j <= order; ++j) {
    //                     size_t jeh = iEH(j);
    //                     vert += coeff_matrices[l].reyy(ieh,jeh) * (b*double(j)-ktran) * H[jeh];
    //                 }
    //                 vert /= k0;
    //                 sum += real(vert * conj(vert));
    //             }
    //         }
    //     } else if (polarization != E_LONG) {
    //         if (symmetric()) {
    //             for (int i = 0; i <= order; ++i) {
    //                 vert = 0.;
    //                 for (int j = 0; j <= order; ++j)
    //                     vert -= coeff_matrices[l].reyy(i,j) * (beta * H[iHx(j)] + (b*double(j)-ktran) * H[iHz(j)]);
    //                 vert /= k0;
    //                 sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
    //             }
    //         } else {
    //             for (int i = -order; i <= order; ++i) {
    //                 vert = 0.;
    //                 for (int j = -order; j <= order; ++j)
    //                     vert -= coeff_matrices[l].reyy(iEH(i),iEH(j)) * (beta * H[iHx(j)] + (b*double(j)-ktran) * H[iHz(j)]);
    //                 vert /= k0;
    //                 sum += real(vert * conj(vert));
    //             }
    //         }
    //     }
    // } else { // which_field == FIELD_H
    //     if (polarization == E_LONG) {  // polarization == H_TRAN
    //         if (symmetric()) {
    //             for (int i = 0; i <= order; ++i) {
    //                 if (periodic)
    //                     vert = - (b*double(i)-ktran) * E[iEH(i)];
    //                 else {
    //                     vert = 0.;
    //                     for (int j = 0; j <= order; ++j)
    //                         vert -= coeff_matrix_rmyy(i,j) * (b*double(j)-ktran) * E[iEH(j)];
    //                 }
    //                 vert /= k0;
    //                 sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
    //             }
    //         } else {
    //             for (int i = -order; i <= order; ++i) {
    //                 if (periodic)
    //                     vert = - (b*double(i)-ktran) * E[iEH(i)];
    //                 else {
    //                     vert = 0.;
    //                     for (int j = -order; j <= order; ++j)
    //                         vert -= coeff_matrix_rmyy(i,j) * (b*double(j)-ktran) * E[iEH(j)];
    //                 }
    //                 vert /= k0;
    //                 sum += real(vert * conj(vert));
    //             }
    //         }
    //     } else if (polarization != E_TRAN) {
    //         for (int i = symmetric()? 0 : -order; i <= order; ++i) {
    //             if (symmetric()) {
    //                 if (periodic)
    //                     vert = (beta * E[iEx(i)] - (b*double(i)-ktran) * E[iEz(i)]);
    //                 else {
    //                     vert = 0.;
    //                     for (int j = 0; j <= order; ++j)
    //                         vert += coeff_matrix_rmyy(i,j) * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
    //                 }
    //                 vert /= k0;
    //                 sum += ((i == 0)? 1. : 2.) * real(vert * conj(vert));
    //             } else {
    //                 if (periodic)
    //                     vert = (beta * E[iEx(i)] - (b*double(i)-ktran) * E[iEz(i)]);
    //                 else {
    //                     vert = 0.;
    //                     for (int j = -order; j <= order; ++j)
    //                         vert += coeff_matrix_rmyy(i,j) * (beta * E[iEx(j)] - (b*double(j)-ktran) * E[iEz(j)]);
    //                 }
    //                 vert /= k0;
    //                 sum += real(vert * conj(vert));
    //             }
    //         }
    //     }
    // }

    // double L = (right-left) * (symmetric()? 2. : 1.);
    // if (field == FIELD_E) {
    //     if (symmetric()) {
    //         for (dcomplex e: E) sum += 2. * real(e * conj(e));
    //         if (separated()) {
    //             size_t i = iEH(0);
    //             sum -= real(E[i] * conj(E[i]));
    //         } else {
    //             size_t ix = iEx(0), iz = iEz(0);
    //             sum -= real(E[ix] * conj(E[ix])) + real(E[iz] * conj(E[iz]));
    //         }
    //     } else {
    //         for (dcomplex e: E) sum += real(e * conj(e));
    //     }
    // } else {
    //     if (symmetric()) {
    //         for (dcomplex h: H) sum += 2. * real(h * conj(h));
    //         if (separated()) {
    //             size_t i = iEH(0);
    //             sum -= real(H[i] * conj(H[i]));
    //         } else {
    //             size_t ix = iHx(0), iz = iHz(0);
    //             sum -= real(H[ix] * conj(H[ix])) + real(H[iz] * conj(H[iz]));
    //         }
    //     } else {
    //         for (dcomplex e: H) sum += real(e * conj(e));
    //     }
    // }
    // return 0.5 * L * sum;
}

}}}  // namespace plask::optical::slab
