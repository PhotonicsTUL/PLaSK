#include "expansion2d.h"
#include "solver2d.h"
#include "../meshadapter.h"

#define SOLVER static_cast<FourierSolver2D*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionPW2D::ExpansionPW2D(FourierSolver2D* solver): Expansion(solver), initialized(false),
    symmetry(E_UNSPECIFIED), polarization(E_UNSPECIFIED) {}

size_t ExpansionPW2D::lcount() const {
    return SOLVER->getLayersPoints().size();
}

void ExpansionPW2D::init()
{
    auto geometry = SOLVER->getGeometry();

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    left = geometry->getChild()->getBoundingBox().lower[0];
    right = geometry->getChild()->getBoundingBox().upper[0];

    size_t refine = SOLVER->refine, M;
    if (refine == 0) refine = 1;

    if (symmetry != E_UNSPECIFIED && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
        throw BadInput(solver->getId(), "Symmetry not allowed for asymmetric() structure");

    if (geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN)) {
        if (right <= 0) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0) throw BadMesh(SOLVER->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric()) left = -right;
    }

    if (!periodic) {
        // Add PMLs
        if (!symmetric()) left -= SOLVER->pml.size + SOLVER->pml.shift;
        right += SOLVER->pml.size + SOLVER->pml.shift;
    }

    double L;
                                                            // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric()) {                                       //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        L = right - left;                                   //  ^ ^ ^ ^ ^
        N = 2 * SOLVER->getSize() + 1;                      // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nN = 4 * SOLVER->getSize() + 1;
        M = refine * nN;                                    // N = 3  nN = 5  refine = 4  M = 20
        double dx = 0.5 * L * (refine-1) / M;               // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        xmesh = RegularAxis(left-dx, right-dx-L/M, M);      //  ^ ^ ^ ^
    } else {                                                // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        L = 2 * right;
        N = SOLVER->getSize() + 1;
        nN = 2 * SOLVER->getSize() + 1;                     // N = 3  nN = 5  refine = 4  M = 20
        M = refine * nN;                                    // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        double dx = 0.25 * L / M;                           //  ^ ^ ^ ^
        xmesh = RegularAxis(left + dx, right - dx, M);      // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    }

    SOLVER->writelog(LOG_DETAIL, "Creating%3%%4% expansion with %1% plane-waves (matrix size: %2%)",
                     N, matrixSize(), symmetric()?" symmetric":"", separated()?" separated":"");

    matFFT = FFT::Forward1D(4, nN, symmetric()? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE);

    // Compute permeability coefficients
    mag.reset(nN, Tensor2<dcomplex>(0.));
    if (periodic) {
        mag[0].c00 = 1.; mag[0].c11 = 1.; // constant 1
    } else {
        // Add PMLs
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure width: %1%um)", L);
        double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
        if (symmetric()) pil = 0;
        else pil = std::lower_bound(xmesh.begin(), xmesh.end(), pl) - xmesh.begin();
        pir = std::lower_bound(xmesh.begin(), xmesh.end(), pr) - xmesh.begin();
        std::fill(mag.begin(), mag.end(), Tensor2<dcomplex>(0.));
        for (size_t i = 0; i != nN; ++i) {
            for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
                dcomplex sy = 1.;
                if (j < pil) {
                    double h = (pl - xmesh[j]) / SOLVER->pml.size;
                    sy = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
                } else if (j > pir) {
                    double h = (xmesh[j] - pr) / SOLVER->pml.size;
                    sy = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
                }
                mag[i] += Tensor2<dcomplex>(sy, 1./sy);
            }
            mag[i] /= refine;
        }
        // Compute FFT
        FFT::Forward1D(2, nN, symmetric()? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(mag.data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI / L; bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN; ++i) {
                int k = i; if (k > nN/2) k -= nN;
                mag[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = lcount();
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);

    initialized = true;
}

void ExpansionPW2D::reset() {
    coeffs.clear();
    initialized = false;
}

void ExpansionPW2D::layerMaterialCoefficients(size_t l)
{
    if (isnan(real(SOLVER->getWavelength())) || isnan(imag(SOLVER->getWavelength())))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();
    const OrderedAxis& axis1 = SOLVER->getLayerPoints(l);

    size_t refine = SOLVER->refine;
    if (refine == 0) refine = 1;
    size_t M = refine * nN;

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DEBUG, "Getting refractive indices for layer %1% (sampled at %2% points) in thread %3%", l, M, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DEBUG, "Getting refractive indices for layer %1% (sampled at %2% points)", l, M);
    #endif

    auto mesh = make_shared<RectangularMesh<2>>(make_shared<RegularAxis>(xmesh), make_shared<OrderedAxis>(axis1), RectangularMesh<2>::ORDER_01);

    double lambda = real(SOLVER->getWavelength());

    auto temperature = SOLVER->inTemperature(mesh);

    LazyData<double> gain;
    bool gain_connected = SOLVER->inGain.hasProvider(), gain_computed = false;

    double factor = 1. / refine;
    double maty = axis1[0]; // at each point along any vertical axis material is the same
    double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
    Tensor3<dcomplex> refl, refr;
    if (!periodic) {
        double Tl = 0.; for (size_t v = pil * axis1.size(), end = (pil+1) * axis1.size(); v != end; ++v) Tl += temperature[v]; Tl /= axis1.size();
        double Tr = 0.; for (size_t v = pir * axis1.size(), end = (pir+1) * axis1.size(); v != end; ++v) Tr += temperature[v]; Tr /= axis1.size();
        refl = geometry->getMaterial(vec(pl,maty))->NR(lambda, Tl).sqr();
        refr = geometry->getMaterial(vec(pr,maty))->NR(lambda, Tr).sqr();
    }

    // Average material parameters
    coeffs[l].reset(nN, Tensor3<dcomplex>(0.));

    for (size_t i = 0; i != nN; ++i) {
        for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
            auto material = geometry->getMaterial(vec(xmesh[j],maty));
            double T = 0.; for (size_t v = j * axis1.size(), end = (j+1) * axis1.size(); v != end; ++v) T += temperature[v]; T /= axis1.size();
            Tensor3<dcomplex> nr = material->NR(lambda, T);
            if (nr.c01 != 0.) {
                if (symmetric()) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
                if (separated()) throw BadInput(solver->getId(), "Single polarization not allowed for structure with non-diagonal NR tensor");
            }
            if (gain_connected) {
                auto roles = geometry->getRolesAt(vec(xmesh[j],maty));
                if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                    if (!gain_computed) {
                        gain = SOLVER->inGain(mesh, lambda);
                        gain_computed = true;
                    }
                    double g = 0.; for (size_t v = j * axis1.size(), end = (j+1) * axis1.size(); v != end; ++v) g += gain[v];
                    double ni = lambda * g/axis1.size() * (0.25e-7/M_PI);
                    nr.c00.imag(ni); nr.c11.imag(ni); nr.c22.imag(ni); nr.c01.imag(0.);
                }
            }
            nr.sqr_inplace();

            // Add PMLs
            if (!periodic) {
                if (j < pil) {
                    double h = (pl - xmesh[j]) / SOLVER->pml.size;
                    dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
                    nr = Tensor3<dcomplex>(refl.c00*sy, refl.c11/sy, refl.c22*sy);
                } else if (j > pir) {
                    double h = (xmesh[j] - pr) / SOLVER->pml.size;
                    dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
                    nr = Tensor3<dcomplex>(refr.c00*sy, refr.c11/sy, refr.c22*sy);
                }
            }

            coeffs[l][i] += Tensor3<dcomplex>(nr.c00, nr.c00/(nr.c00*nr.c11-nr.c01*nr.c01), nr.c22, nr.c01);
        }
        coeffs[l][i] *= factor;
        if (coeffs[l][i].c11 != 0. && !isnan(coeffs[l][i].c11.real()) && !isnan(coeffs[l][i].c11.imag()))
            coeffs[l][i].c11 = 1. / coeffs[l][i].c11; // We were averaging inverses of c11 (xx)
        else coeffs[l][i].c11 = 0.;
        if (coeffs[l][i].c22 != 0.)
            coeffs[l][i].c22 = 1. / coeffs[l][i].c22; // We need inverse of c22 (yy)
    }

    // Check if the layer is uniform
    if (periodic) {
        diagonals[l] = true;
        for (size_t i = 1; i != nN; ++i) {
            Tensor3<dcomplex> diff = coeffs[l][i] - coeffs[l][0];
            if (!(is_zero(diff.c00) && is_zero(diff.c11) && is_zero(diff.c22) && is_zero(diff.c01))) {
                diagonals[l] = false;
                break;
            }
        }
    } else
        diagonals[l] = false;

    if (diagonals[l]) {
        solver->writelog(LOG_DETAIL, "Layer %1% is uniform", l);
        for (size_t i = 1; i != nN; ++i) coeffs[l][i] = Tensor3<dcomplex>(0.);
    } else {
        // Perform FFT
        matFFT.execute(reinterpret_cast<dcomplex*>(coeffs[l].data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI / ((right-left) * (symmetric()? 2 : 1)); bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN; ++i) {
                int k = i; if (k > nN/2) k -= nN;
                coeffs[l][i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
}


LazyData<Tensor3<dcomplex>> ExpansionPW2D::getMaterialNR(size_t l, const shared_ptr<const LevelsAdapter::Level> &level, InterpolationMethod interp)
{
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level-> mesh());
    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        if (!symmetric()) {
            return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this,l,dest_mesh](size_t i)->Tensor3<dcomplex>{
                Tensor3<dcomplex> eps(0.);
                for (int k = -int(nN)/2, end = int(nN+1)/2; k != end; ++k) {
                    size_t j = (k>=0)? k : k + nN;
                    eps += coeffs[l][j] * exp(2*M_PI * k * I * (dest_mesh->at(i).c0-left) / (right-left));
                }
                eps.c22 = 1. / eps.c22;
                eps.sqrt_inplace();
                return eps;
            });
        } else {
            return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this,l,dest_mesh](size_t i)->Tensor3<dcomplex>{
                Tensor3<dcomplex> eps = coeffs[l][0];
                for (int k = 1; k != nN; ++k) {
                    eps += 2. * coeffs[l][k] * cos(M_PI * k * dest_mesh->at(i).c0 / (right-left));
                }
                eps.c22 = 1. / eps.c22;
                eps.sqrt_inplace();
                return eps;
            });
        }
    } else {
        DataVector<Tensor3<dcomplex>> params(symmetric()? nN : nN+1);
        std::copy(coeffs[l].begin(), coeffs[l].end(), params.begin());
        FFT::Backward1D(4, nN, symmetric()? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> cmesh = make_shared<RegularAxis>();
        if (symmetric()) {
            double dx = 0.5 * (right-left) / nN;
            cmesh->reset(left + dx, right - dx, nN);
        } else {
            cmesh->reset(left, right, nN+1);
            params[nN] = params[0];
        }
        for (Tensor3<dcomplex>& eps: params) {
            eps.c22 = 1. / eps.c22;
            eps.sqrt_inplace();
        }
        auto src_mesh = make_shared<RectangularMesh<2>>(cmesh, make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
        return interpolate(src_mesh, params, dest_mesh, interp,
                           InterpolationFlags(SOLVER->getGeometry(),
                                              symmetric()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                              InterpolationFlags::Symmetry::NO)
                          );
    }
}


void ExpansionPW2D::getMatrices(size_t l, dcomplex k0, dcomplex beta, dcomplex kx, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);

    int order = SOLVER->getSize();
    dcomplex f = 1. / k0, k02 = k0*k0;
    double b = 2*M_PI / (right-left) * (symmetric()? 0.5 : 1.0);

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
                    dcomplex gi = b * double(i) - kx;
                    size_t ie = iE(i), ih = iH(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
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
                    dcomplex gi = b * double(i) - kx;
                    size_t ie = iE(i), ih = iH(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
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
                dcomplex gi = b * double(i) - kx;
                size_t iex = iEx(i), iez = iEz(i), ihx = iHx(i), ihz = iHz(i);
                for (int j = -order; j <= order; ++j) {
                    int ij = i-j;   dcomplex gj = b * double(j) - kx;
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
    if (field_params.method == INTERPOLATION_DEFAULT) field_params.method = INTERPOLATION_SPLINE;
    if (symmetric()) {
        field.reset(N);
        Component sym = (field_params.which == FieldParams::E)? symmetry : Component(2-symmetry);
        if (field_params.method != INTERPOLATION_FOURIER) {
            fft_x = FFT::Backward1D(1, N, FFT::Symmetry(sym), 3);    // tran
            fft_yz = FFT::Backward1D(1, N, FFT::Symmetry(2-sym), 3); // long
        }
    } else {
        field.reset(N + 1);
        if (field_params.method != INTERPOLATION_FOURIER)
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

DataVector<const Vec<3,dcomplex>> ExpansionPW2D::getField(size_t l, const shared_ptr<const typename LevelsAdapter::Level> &level, const cvector& E, const cvector& H)
{
    Component sym = (field_params.which == FieldParams::E)? symmetry : Component(2-symmetry);

    const dcomplex beta = field_params.klong;
    const dcomplex kx = field_params.ktran;

    int order = SOLVER->getSize();
    double b = 2*M_PI / (right-left) * (symmetric()? 0.5 : 1.0);
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double vpos = level->vpos();

    int dx = (symmetric() && field_params.method != INTERPOLATION_FOURIER && sym != E_TRAN)? 1 : 0; // 1 for sin expansion of tran component
    int dz = (symmetric() && field_params.method != INTERPOLATION_FOURIER && sym != E_LONG)? 1 : 0; // 1 for sin expansion of long component

    if (field_params.which == FieldParams::E) {
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
                                field[iE(i)-dz].vert() += iepsyy(l,i-j) * (b*double(j)-kx) * H[iH(j)];
                        }
                        field[iE(i)-dz].vert() /= field_params.k0;
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
                            field[iE(i)-dz].vert() -= iepsyy(l,i-j) * (beta * H[iHx(i)] + (b*double(j)-kx) * H[iHz(j)]);
                    }
                    field[iE(i)-dz].vert() /= field_params.k0;
                }
            }
        }
    } else { // field_params.which == FieldParams::H
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
                                field[iH(i)-dz].vert() -= imuyy(l,i-j) * (b*double(j)-kx) * E[iE(j)];
                        }
                        field[iH(i)-dz].vert() /= field_params.k0;
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
                            field[iE(i)-dz].vert() += imuyy(l,i-j) * (beta * E[iEx(j)] - (b*double(j)-kx) * E[iEz(j)]);
                    }
                    field[iH(i)].vert() /= field_params.k0;
                }
            }
        }
   }

    if (dx) { field[field.size()-1].tran() = 0.; }
    if (dz) { field[field.size()-1].lon() = 0.; field[field.size()-1].vert() = 0.; }

    if (field_params.method == INTERPOLATION_FOURIER) {
        DataVector<Vec<3,dcomplex>> result(dest_mesh->size());
        double L = right - left;
        if (!symmetric()) {
            dcomplex B = 2*M_PI * I / L;
            dcomplex ikx = I * kx;
            result.reset(dest_mesh->size(), Vec<3,dcomplex>(0.,0.,0.));
            for (int k = -order; k <= order; ++k) {
                size_t j = (k>=0)? k : k + N;
                dcomplex G = B * double(k) - ikx;
                for (size_t i = 0; i != dest_mesh->size(); ++i) {
                    result[i] += field[j] * exp(G * (dest_mesh->at(i)[0]-left));
                }
            }
        } else {
            double B = M_PI / L;
            result.reset(dest_mesh->size());
            for (size_t i = 0; i != dest_mesh->size(); ++i) {
                result[i] = field[0];
                for (int k = 1; k <= order; ++k) {
                    double cs =  2. * cos(B * k * dest_mesh->at(i)[0]);
                    double sn =  2. * sin(B * k * dest_mesh->at(i)[0]);
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
            double dx = 0.5 * (right-left) / N;
            auto src_mesh = make_shared<RectangularMesh<2>>(make_shared<RegularAxis>(left+dx, right-dx, field.size()), make_shared<RegularAxis>(vpos, vpos, 1));
            return interpolate(src_mesh, field, dest_mesh, field_params.method, 
                               InterpolationFlags(SOLVER->getGeometry(),
                                    (sym == E_TRAN)? InterpolationFlags::Symmetry::NPN : InterpolationFlags::Symmetry::PNP,
                                    InterpolationFlags::Symmetry::NO), 
                                    false);
        } else {
            fft_x.execute(reinterpret_cast<dcomplex*>(field.data()));
            field[N] = field[0];
            auto src_mesh = make_shared<RectangularMesh<2>>(make_shared<RegularAxis>(left, right, field.size()), make_shared<RegularAxis>(vpos, vpos, 1));
            auto result = interpolate(src_mesh, field, dest_mesh, field_params.method,
                                      InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::NO),
                                      false).claim();
            dcomplex ikx = I * kx;
            for (size_t i = 0; i != dest_mesh->size(); ++i)
                result[i] *= exp(- ikx * dest_mesh->at(i).c0);
            return result;
        }
    }
}


}}} // namespace plask
