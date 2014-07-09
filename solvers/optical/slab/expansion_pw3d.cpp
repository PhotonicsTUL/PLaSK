#include "expansion_pw3d.h"
#include "fourier_reflection_3d.h"
#include "mesh_adapter.h"

#define SOLVER static_cast<FourierReflection3D*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionPW3D::ExpansionPW3D(FourierReflection3D* solver): Expansion(solver), initialized(false),
    symmetry_long(E_UNSPECIFIED), symmetry_tran(E_UNSPECIFIED) {}

size_t ExpansionPW3D::lcount() const {
    return SOLVER->getLayersPoints().size();
}

void ExpansionPW3D::init()
{
    auto geometry = SOLVER->getGeometry();

    periodic_long = geometry->isPeriodic(Geometry3D::DIRECTION_LONG);
    periodic_tran = geometry->isPeriodic(Geometry3D::DIRECTION_TRAN);

    back = geometry->getChild()->getBoundingBox().lower[0];
    front = geometry->getChild()->getBoundingBox().upper[0];
    left = geometry->getChild()->getBoundingBox().lower[1];
    right = geometry->getChild()->getBoundingBox().upper[1];

    size_t refl = SOLVER->refine_long, reft = SOLVER->refine_tran, Ml, Mt;
    if (refl == 0) refl = 1;  if (reft == 0) reft = 1;

    symmetric_long = symmetric_tran = false;
    if (symmetry_long != E_UNSPECIFIED) {
        if (!geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
            throw BadInput(solver->getId(), "Longitudinal symmetry not allowed for asymmetric structure");
        symmetric_long = true;
    }
    if (symmetry_tran != E_UNSPECIFIED) {
        if (!geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
            throw BadInput(solver->getId(), "Transverse symmetry not allowed for asymmetric structure");
        symmetric_tran = true;
    }

    if (geometry->isSymmetric(Geometry3D::DIRECTION_LONG)) {
        if (front <= 0) {
            back = -back; front = -front;
            std::swap(back, front);
        }
        if (back != 0) throw BadMesh(SOLVER->getId(), "Longitudinally symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_long) back = -front;
    }
    if (geometry->isSymmetric(Geometry3D::DIRECTION_TRAN)) {
        if (right <= 0) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0) throw BadMesh(SOLVER->getId(), "Transversely symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_tran) left = -right;
    }

    if (!periodic_long) {
        // Add PMLs
        if (!symmetric_long) back -= SOLVER->pml_long.size + SOLVER->pml_long.shift;
        front += SOLVER->pml_long.size + SOLVER->pml_long.shift;
    }
    if (!periodic_tran) {
        // Add PMLs
        if (!symmetric_tran) left -= SOLVER->pml_tran.size + SOLVER->pml_tran.shift;
        right += SOLVER->pml_tran.size + SOLVER->pml_tran.shift;
    }

    double Ll, Lt;

    if (!symmetric_long) {
        Ll = front - back;
        Nl = 2 * SOLVER->getLongSize() + 1;
        nNl = 4 * SOLVER->getLongSize() + 1;
        Ml = refl * nNl;
        double dx = 0.5 * Ll * (refl-1) / Ml;
        long_mesh = RegularAxis(back-dx, front-dx-Ll/Ml, Ml);
    } else {
        Ll = 2 * front;
        Nl = SOLVER->getLongSize() + 1;
        nNl = 2 * SOLVER->getLongSize() + 1;
        Ml = refl * nNl;
        double dx = 0.25 * Ll / Ml;
        long_mesh = RegularAxis(back + dx, front - dx, Ml);
    }                                                           // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric_tran) {                                      //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        Lt = right - left;                                      //  ^ ^ ^ ^ ^
        Nt = 2 * SOLVER->getTranSize() + 1;                     // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nNt = 4 * SOLVER->getTranSize() + 1;
        Mt = reft * nNt;                                        // N = 3  nN = 5  refine = 4  M = 20
        double dx = 0.5 * Lt * (reft-1) / Mt;                   // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        tran_mesh = RegularAxis(left-dx, right-dx-Lt/Mt, Mt);   //  ^ ^ ^ ^
    } else {                                                    // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        Lt = 2 * right;
        Nt = SOLVER->getTranSize() + 1;
        nNt = 2 * SOLVER->getTranSize() + 1;                    // N = 3  nN = 5  refine = 4  M = 20
        Mt = reft * nNt;                                        // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        double dx = 0.25 * Lt / Mt;                             //  ^ ^ ^ ^
        tran_mesh = RegularAxis(left + dx, right - dx, Mt);     // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    }

    SOLVER->writelog(LOG_DETAIL, "Creating expansion%4% with %1%x%2% plane-waves (matrix size: %3%)", Nl, Nt, matrixSize(),
                     (!symmetric_long && !symmetric_tran)? "" :
                     (symmetric_long && symmetric_tran)? " symmetric in longitudinal and transverse directions" :
                     (!symmetric_long && symmetric_tran)? " symmetric in transverse direction" : " symmetric in longitudinal direction"
                    );

    matFFT = FFT::Forward2D(4, nNl, nNt, symmetric_long? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE, symmetric_tran? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE);

    // Compute permeability coefficients
    mag_long.reset(nNl, Tensor2<dcomplex>(0.));
    mag_tran.reset(nNt, Tensor2<dcomplex>(0.));
    if (!periodic_long || !periodic_tran) {
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure dimensions: %1%um x %2%um)", Ll, Lt);
    }
    if (periodic_long) {
        mag_long[0].c00 = 1.; mag_long[0].c11 = 1.; // constant 1
    } else {
        double pb = back + SOLVER->pml_long.size, pf = front - SOLVER->pml_long.size;
        if (symmetric_long) pib = 0;
        else pib = std::lower_bound(long_mesh.begin(), long_mesh.end(), pb) - long_mesh.begin();
        pif = std::lower_bound(long_mesh.begin(), long_mesh.end(), pf) - long_mesh.begin();
        for (size_t i = 0; i != nNl; ++i) {
            for (size_t j = refl*i, end = refl*(i+1); j != end; ++j) {
                dcomplex s = 1.;
                if (j < pib) {
                    double h = (pb - long_mesh[j]) / SOLVER->pml_long.size;
                    s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                } else if (j > pif) {
                    double h = (long_mesh[j] - pf) / SOLVER->pml_long.size;
                    s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                }
                mag_long[i] += Tensor2<dcomplex>(s, 1./s);
            }
            mag_long[i] /= refl;
        }
        // Compute FFT
        FFT::Forward1D(2, nNl, symmetric_long? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(mag_long.data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI*M_PI / Ll / Lt;   // (2π/L)² / 4
            for (size_t i = 0; i != nNl; ++i) {
                int k = i; if (k > nNl/2) k -= nNl;
                mag_long[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
    if (periodic_tran) {
        mag_tran[0].c00 = 1.; mag_tran[0].c11 = 1.; // constant 1
    } else {
        double pb = back + SOLVER->pml_tran.size, pf = front - SOLVER->pml_tran.size;
        if (symmetric_tran) pil = 0;
        else pil = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pb) - tran_mesh.begin();
        pir = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pf) - tran_mesh.begin();
        for (size_t i = 0; i != nNt; ++i) {
            for (size_t j = reft*i, end = reft*(i+1); j != end; ++j) {
                dcomplex s = 1.;
                if (j < pil) {
                    double h = (pb - tran_mesh[j]) / SOLVER->pml_tran.size;
                    s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                } else if (j > pir) {
                    double h = (tran_mesh[j] - pf) / SOLVER->pml_tran.size;
                    s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                }
                mag_tran[i] += Tensor2<dcomplex>(s, 1./s);
            }
            mag_tran[i] /= reft;
        }
        // Compute FFT
        FFT::Forward1D(2, nNt, symmetric_tran? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(mag_tran.data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI*M_PI / Ll / Lt;   // (2π/L)² / 4
            for (size_t i = 0; i != nNt; ++i) {
                int k = i; if (k > nNt/2) k -= nNt;
                mag_tran[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = lcount();
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);

    initialized = true;
}

void ExpansionPW3D::free() {
    coeffs.clear();
    initialized = false;
}

template <typename T1, typename T2>
inline static Tensor3<decltype(T1()*T2())> commutator(const Tensor3<T1>& A, const Tensor3<T2>& B) {
    return Tensor3<decltype(T1()*T2())>(
        A.c00 * B.c00 + A.c01 * B.c01,
        A.c01 * B.c01 + A.c11 * B.c11,
        A.c22 * B.c22,
        0.5 * ((A.c00 + A.c11) * B.c01 + A.c01 * (B.c00 + B.c11))
    );
}

void ExpansionPW3D::layerMaterialCoefficients(size_t l)
{
    if (isnan(real(SOLVER->getWavelength())) || isnan(imag(SOLVER->getWavelength())))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();
    const OrderedAxis& axis2 = SOLVER->getLayerPoints(l);

    const double Lt = right - left, Ll = front - back;
    const size_t refl = (SOLVER->refine_long)? SOLVER->refine_long : 1,
                 reft = (SOLVER->refine_tran)? SOLVER->refine_tran : 1;
    const size_t Ml = refl * nNl,  Mt = reft * nNt;
    size_t nN = nNl * nNt;
    const double normlim = min(Ll/nNl, Lt/nNt) * 1e-9;

    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer %1% (sampled at %2%x%3% points)", l, Ml, Mt);

    auto mesh = make_shared<RectangularMesh<3>>
                           (make_shared<RegularAxis>(long_mesh),
                            make_shared<RegularAxis>(tran_mesh),
                            make_shared<OrderedAxis>(axis2),
                            RectangularMesh<3>::ORDER_102);
    double matv = axis2[0]; // at each point along any vertical axis material is the same

    double lambda = real(SOLVER->getWavelength());

    auto temperature = SOLVER->inTemperature(mesh);

    LazyData<double> gain;
    bool have_gain = false;
    if (SOLVER->inGain.hasProvider()) {
        gain = SOLVER->inGain(mesh, lambda);
        have_gain = true;
    }

    // Average material parameters
    coeffs[l].reset(nN, Tensor3<dcomplex>(0.));

    DataVector<Tensor3<dcomplex>> cell(refl*reft);
    double nfact = 1. / cell.size();

    double pb = back + SOLVER->pml_long.size, pf = front - SOLVER->pml_long.size;
    double pl = left + SOLVER->pml_tran.size, pr = right - SOLVER->pml_tran.size;

    for (size_t it = 0; it != nNt; ++it) {
        size_t tbegin = reft * it; size_t tend = tbegin + reft;
        double tran0 = 0.5 * (tran_mesh[tbegin] + tran_mesh[tend-1]);

        for (size_t il = 0; il != nNl; ++il) {
            size_t lbegin = refl * il; size_t lend = lbegin + refl;
            double long0 = 0.5 * (long_mesh[lbegin] + long_mesh[lend-1]);

            // Store epsilons for a single cell and compute surface normal
            Vec<2> norm(0.,0.);
            for (size_t t = tbegin, j = 0; t != tend; ++t) {
                for (size_t l = lbegin; l != lend; ++l, ++j) {
                    auto material = geometry->getMaterial(vec(long_mesh[l], tran_mesh[t], matv));
                    double T = 0.; // average temperature in all vertical points
                    for (size_t v = mesh->index(l, t, 0), end = mesh->index(l, t, axis2.size()); v != end; ++v) T += temperature[v];
                    T /= axis2.size();
                    cell[j] = material->NR(lambda, T);
                    if (cell[j].c01 != 0.) {
                        if (symmetric_long || symmetric_tran) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
                    }
                    if (have_gain) {
                        auto roles = geometry->getRolesAt(vec(long_mesh[l], tran_mesh[t], matv));
                        if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                            double g = 0.; // average gain in all vertical points
                            for (size_t v = mesh->index(l, t, 0) * axis2.size(), end = mesh->index(l, t, axis2.size())+1; v != end; ++v) g += gain[v];
                            double ni = lambda * g/axis2.size() * (0.25e-7/M_PI);
                            cell[j].c00.imag(ni);
                            cell[j].c11.imag(ni);
                            cell[j].c22.imag(ni);
                            cell[j].c01.imag(0.);
                        }
                    }
                    auto& eps = cell[j];
                    eps.sqr_inplace();  // make epsilon from NR

                    // Add PMLs
                    if (!periodic_long) {
                        dcomplex s = 1.;
                        if (l < pib) {
                            double h = (pb - long_mesh[l]) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                        } else if (l > pif) {
                            double h = (long_mesh[l] - pf) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                        }
                        cell[j].c00 *= 1./s;
                        cell[j].c11 *= s;
                        cell[j].c22 *= s;
                    }
                    if (!periodic_tran) {
                        dcomplex s = 1.;
                        if (t < pil) {
                            double h = (pl - tran_mesh[t]) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                        } else if (t > pir) {
                            double h = (tran_mesh[t] - pr) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                        }
                        cell[j].c00 *= s;
                        cell[j].c11 *= 1./s;
                        cell[j].c22 *= s;
                    }

                    norm += (real(eps.c00) + real(eps.c11)) * vec(long_mesh[l] - long0, tran_mesh[t] - tran0);
                }
            }

            double a = abs(norm);
            auto& eps = coeffs[l][nNl * it + il];
            if (a < normlim) {
                // coeffs[l][nNl * it + il] = Tensor3<dcomplex>(0.); // just for testing
                // Nothing to average
                eps = cell[cell.size() / 2];
            } else {
                // eps = Tensor3<dcomplex>(norm.c0/a, norm.c1/a, 0.); // just for testing

                // Compute avg(eps) and avg(eps**(-1))
                Tensor3<dcomplex> ieps(0.);
                for (size_t t = tbegin, j = 0; t != tend; ++t) {
                    for (size_t l = lbegin; l != lend; ++l, ++j) {
                        eps += cell[j];
                        ieps += cell[j].inv();
                    }
                }
                eps *= nfact;
                ieps *= nfact;

                // Average permittivity tensor according to:
                // [ S. G. Johnson and J. D. Joannopoulos, Opt. Express, vol. 8, pp. 173-190 (2001) ]
                norm /= a;
                Tensor3<double> P(norm.c0*norm.c0, norm.c1*norm.c1, 0., norm.c0*norm.c1);
                Tensor3<double> P1(1.-P.c00, 1.-P.c11, 1., -P.c01);
                eps = commutator(P, ieps.inv()) + commutator(P1, eps);
            }
            eps.c22 = 1./eps.c22;
        }
    }

    // Check if the layer is uniform
    if (periodic_tran && periodic_long) {
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
            double bb4l = M_PI / ((front-back) * (symmetric_long? 2 : 1)); bb4l *= bb4l; // (2π/Ll)² / 4
            double bb4t = M_PI / ((right-left) * (symmetric_tran? 2 : 1)); bb4t *= bb4t; // (2π/Lt)² / 4
            for (size_t it = 0; it != nNl; ++it) {
                int kt = it; if (kt > nNt/2) kt -= nNt;
                for (size_t il = 0; il != nNl; ++il) {
                    int kl = il; if (kl > nNl/2) kl -= nNl;
                    coeffs[l][nNl*it+il] *= exp(-SOLVER->smooth * (bb4l * kl*kl + bb4t * kt*kt));
                }
            }
        }
    }
}


DataVector<const Tensor3<dcomplex>> ExpansionPW3D::getMaterialNR(size_t lay, OrderedAxis lmesh, OrderedAxis tmesh, InterpolationMethod interp)
{
    DataVector<Tensor3<dcomplex>> result(lmesh.size() * tmesh.size(), Tensor3<dcomplex>(0.));
    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        const double Lt = right - left, Ll = front - back;
        const int endt = int(nNt+1)/2, endl = int(nNl+1)/2;
        for (int kt = -int(nNt)/2; kt != endt; ++kt) {
            size_t t = (kt >= 0)? kt : (symmetric_tran)? -kt : kt + nNt;
            for (int kl = -int(nNl)/2; kl != endl; ++kl) {
                size_t l = (kl >= 0)? kl : (symmetric_long)? -kl : kl + nNl;
                for (size_t it = 0; it != tmesh.size(); ++it) {
                    const double phast = kt * (tmesh[it]-left) / Lt;
                    size_t offset = lmesh.size()*it;
                    for (size_t il = 0; il != lmesh.size(); ++il) {
                        result[offset+il] += coeffs[lay][nNl*t+l] * exp(2*M_PI * I * (kl*(lmesh[il]-back)/ Ll + phast));
                    }
                }
            }
        }
    } else {
        size_t nl = symmetric_long? nNl : nNl+1, nt = symmetric_tran? nNt : nNt+1;
        DataVector<Tensor3<dcomplex>> params(nl * nt);
        for (size_t t = 0; t != nNt; ++t) {
            size_t op = nl * t, oc = nNl * t;
            for (size_t l = 0; l != nNl; ++l) {
                params[op+l] = coeffs[lay][oc+l];
            }
        }
        FFT::Backward2D(4, nNl, nNt,
                        symmetric_long? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE,
                        symmetric_tran? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE,
                        0, nl
                       )
            .execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> lcmesh = make_shared<RegularAxis>(), tcmesh = make_shared<RegularAxis>();
        if (symmetric_long) {
            double dx = 0.5 * (front-back) / nNl;
            lcmesh->reset(back+dx, front-dx, nNl);
        } else {
            lcmesh->reset(back, front, nNl+1);
            for (size_t l = 0, last = nl*nNt; l != nNl; ++l) params[last+l] = params[l];
        }
        if (symmetric_tran) {
            double dx = 0.5 * (right-left) / nNt;
            tcmesh->reset(left+dx, right-dx, nNt);
        } else {
            tcmesh->reset(left, right, nNt+1);
            for (size_t t = 0, end = nl*nt; t != end; t += nl) params[nNl+t] = params[t];
        }
        auto src_mesh = make_shared<RectangularMesh<3>>(lcmesh, tcmesh, make_shared<RegularAxis>(0,0,1), RectangularMesh<3>::ORDER_210);
        auto dst_mesh = make_shared<RectangularMesh<3>>(make_shared<OrderedAxis>(std::move(lmesh)), make_shared<OrderedAxis>(std::move(tmesh)),
                                    shared_ptr<OrderedAxis>(new OrderedAxis{0}), RectangularMesh<3>::ORDER_210);
        const bool ignore_symmetry[3] = { !symmetric_long, !symmetric_tran, false };
        result = interpolate(src_mesh, params, make_shared<const WrappedMesh<3>>(dst_mesh, SOLVER->getGeometry(), ignore_symmetry), interp).claim();
    }
    for (Tensor3<dcomplex>& eps: result) {
        eps.c22 = 1. / eps.c22;
        eps.sqrt_inplace();
    }
    return result;
}



void ExpansionPW3D::getMatrices(size_t lay, dcomplex k0, dcomplex klong, dcomplex ktran, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);

    int ordl = SOLVER->getLongSize(), ordt = SOLVER->getTranSize();

    char symx = int(symmetric_long)-1, symy = int(symmetric_tran)-1;
    // +1: Ex+, Ey-, Hx-, Hy+
    //  0: no symmetry
    // -1: Ex-, Ey+, Hx+, Hy-

    double Gx = 2.*M_PI / (front-back) * (symx ? 0.5 : 1.),
           Gy = 2.*M_PI / (right-left) * (symy ? 0.5 : 1.);

    dcomplex k02 = k0 * k0;

    size_t N = (symx ? ordl+1 : 2*ordl+1) * (symy ? ordt+1 : 2*ordt+1);
    std::fill_n(RE.data(), 4*N*N, dcomplex(0.));
    std::fill_n(RH.data(), 4*N*N, dcomplex(0.));

    for (int iy = (symy ? 0 : -ordt); iy <= ordt; ++iy) {

        dcomplex gy = iy * Gy - ktran;

        for (int ix = (symx ? 0 : -ordl); ix <= ordl; ++ix) {

            dcomplex gx = ix * Gx - klong;

            size_t iex = iEx(ix, iy), iey = iEy(ix, iy);
            size_t ihx = iHx(ix, iy), ihy = iHy(ix, iy);

            for (int jy = -ordt; jy <= ordt; ++jy) {

                dcomplex py = jy * Gy - ktran;

                int ijy = iy - jy; if (symy && ijy < 0) ijy = - ijy;

                for (int jx = -ordl; jx <= ordl; ++jx) {

                    dcomplex px = jx * Gx - klong;

                    int ijx = ix - jx; if (symx && ijx < 0) ijx = - ijx;

                    size_t jex = iEx(jx, jy), jey = iEy(jx, jy);
                    size_t jhx = iHx(jx, jy), jhy = iHy(jx, jy);

                    double fx = 1., fy = 1.;
                    if (symx && jx < 0) { fx *= symx; fy *= -symx; }
                    if (symy && jy < 0) { fx *= symy; fy *= -symy; }

                    dcomplex ieps = iepszz(lay, ijx, ijy);
                    RH(iex,jhy) += - fx * gx * px * ieps + k02 * muyy(lay, ijx, ijy);
                    RH(iex,jhx) += - fy * gx * py * ieps;
                    RH(iey,jhy) += - fx * gy * px * ieps;
                    RH(iey,jhx) += - fy * gy * py * ieps + k02 * muxx(lay, ijx, ijy);

                    dcomplex imu = imuzz(lay, ijx, ijy);
                    RE(ihy,jex) += - fx * gy * py * imu + k02 * epsxx(lay, ijx, ijy);
                    RE(ihy,jey) +=   fy * gy * px * imu + k02 * epsxy(lay, ijx, ijy);
                    RE(ihx,jex) +=   fx * gx * py * imu + k02 * epsyx(lay, ijx, ijy);
                    RE(ihx,jey) += - fy * gx * px * imu + k02 * epsyy(lay, ijx, ijy);
                }
            }
        }
    }
}


// void ExpansionPW3D::prepareField()
// {
//     field.reset(N + (symmetric? 0 : 1));
//     Component sym = (field_params.which == FieldParams::E)? symmetry : Component(2-symmetry);
//     fft_x = FFT::Backward1D(1, N, (sym==E_TRAN)? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_ODD, 3);
//     fft_yz = FFT::Backward1D(1, N, (sym==E_TRAN)? FFT::SYMMETRY_ODD : FFT::SYMMETRY_EVEN, 3);
// }
//
// void ExpansionPW3D::cleanupField()
// {
//     field.reset();
//     fft_x = FFT::Backward1D();
//     fft_yz = FFT::Backward1D();
// }
//
// // TODO fields must be carefully verified

DataVector<const Vec<3, dcomplex> > ExpansionPW3D::getField(size_t l, const shared_ptr<const Mesh> &dst_mesh, const cvector& E, const cvector& H)
{
//     Component sym = (field_params.which == FieldParams::E)? symmetry : Component(2-symmetry);
//
//     const dcomplex klong = field_params.klong;
//     const dcomplex kx = field_params.ktran;
//
//     int order = SOLVER->getSize();
//     double b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);
//     assert(dynamic_cast<const LevelMeshAdapter<2>*>(&dst_mesh));
//     const MeshD<2>& dest_mesh = static_cast<const MeshD<2>&>(dst_mesh);
//     double vpos = static_cast<const LevelMeshAdapter<2>&>(dst_mesh).vpos();
//
//     int dt = (symmetric && field_params.method != INTERPOLATION_FOURIER && sym != E_TRAN)? 1 : 0;
//     int dl = (symmetric && field_params.method != INTERPOLATION_FOURIER && sym != E_LONG)? 1 : 0;
//
//     if (field_params.which == FieldParams::E) {
//         if (separated) {
//             if (polarization == E_TRAN) {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iE(i)].lon() = field[iE(i)].vert() = 0.;
//                     if (iE(i) != 0 || !dt) field[iE(i)-dt].tran() = E[iE(i)];
//                 }
//             } else {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iE(i)].tran() = 0.;
//                     if (iE(i) != 0 || !dl) {
//                         field[iE(i)-dl].lon() = - E[iE(i)];
//                         field[iE(i)-dl].vert() = - iepsyy(l, i) * klong * H[iH(i)] / field_params.k0;
//                     }
//                 }
//             }
//         } else {
//             for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                 if (iE(i) != 0 || !dt) field[iE(i)-dt].tran() = E[iEx(i)];
//                 if (iE(i) != 0 || !dl) {
//                     field[iE(i)-dl].lon() = - E[iEz(i)];
//                     field[iE(i)-dl].vert() = - iepsyy(l, i) * klong * H[iHx(i)];
//                     if (symmetric) {
//                         if (sym == E_LONG) {
//                             for (int j = -order; j <= order; ++j)
//                                 field[iE(i)-dl].vert() += iepsyy(l, abs(i-j)) * b*double(j) * H[iHz(abs(j))];
//                         } else {
//                             for (int j = 0; j <= order; ++j)
//                                 field[iE(i)-dl].vert() += iepsyy(l, abs(i-j)) * b*double(j) * H[iHz(j)];
//                             for (int j = -order; j < 0; ++j)
//                                 field[iE(i)-dl].vert() -= iepsyy(l, abs(i-j)) * b*double(j) * H[iHz(-j)];
//                         }
//                     } else {
//                         for (int j = -order; j <= order; ++j)
//                             field[iE(i)-dl].vert() += iepsyy(l, i-j) * (b*double(j)-kx) * H[iHz(j)];
//                     }
//                     field[iE(i)-dl].vert() /= field_params.k0;
//                 }
//             }
//         }
//     } else { // field_params.which == FieldParams::H
//         if (separated) {
//             if (polarization == E_LONG) {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iH(i)].lon() = field[iH(i)].vert() = 0.;
//                     if (iH(i) != 0 || !dt) field[iH(i)- dt].tran() = E[iH(i)];
//                 }
//             } else {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iH(i)].tran() = 0.;
//                     if (iH(i) != 0 || !dl) {
//                         field[iH(i)- dl].lon() = E[iH(i)];
//                         field[iH(i)- dl].vert() = - imuyy(l, i) * klong * E[iE(i)] / field_params.k0;
//                     }
//                 }
//             }
//         } else {
//             for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                 if (iH(i) != 0 || !dt) field[iH(i)- dt].tran() = H[iHx(i)];
//                 if (iH(i) != 0 || !dl) {
//                     field[iH(i)- dl].lon() = H[iHz(i)];
//                     field[iH(i)- dl].vert() = - imuyy(l, i) * klong * H[iEx(i)];
//                     if (symmetric) {
//                         if (sym == E_LONG) {
//                             for (int j = -order; j <= order; ++j)
//                                 field[iE(i)- dl].vert() += imuyy(l, abs(i-j)) * b*double(j) * H[iEz(abs(j))];
//                         } else {
//                             for (int j = 0; j <= order; ++j)
//                                 field[iE(i)- dl].vert() += imuyy(l, abs(i-j)) * b*double(j) * H[iEz(j)];
//                             for (int j = -order; j < 0; ++j)
//                                 field[iE(i)- dl].vert() -= imuyy(l, abs(i-j)) * b*double(j) * H[iEz(-j)];
//                         }
//                     } else {
//                         for (int j = -order; j <= order; ++j)
//                             field[iE(i)- dl].vert() += imuyy(l, i-j) * (b*double(j)-kx) * H[iEz(j)];
//                     }
//                     field[iH(i)].vert() /= field_params.k0;
//                 }
//             }
//         }
//     }
//
//     if (dt) { field[field.size()-1].tran() = 0.; }
//     if (dl) { field[field.size()-1].lon() = 0.; field[field.size()-1].vert() = 0.; }
//
//     if (field_params.method == INTERPOLATION_FOURIER) {
//         DataVector<Vec<3,dcomplex>> result(dest_mesh.size());
//         double L = right - left;
//         if (!symmetric) {
//             result.reset(dest_mesh.size(), Vec<3,dcomplex>(0.,0.,0.));
//             for (int k = -order; k <= order; ++k) {
//                 size_t j = (k>=0)? k : k + N;
//                 for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                     result[i] += field[j] * exp(2*M_PI * k * I * (dest_mesh[i][0]-left) / L);
//                 }
//             }
//         } else {
//             result.reset(dest_mesh.size());
//             for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                 result[i] = field[0];
//                 for (int k = 1; k <= order; ++k) {
//                     double cs =  2. * cos(M_PI * k * dest_mesh[i][0] / L);
//                     double sn =  2. * sin(M_PI * k * dest_mesh[i][0] / L);
//                     if (sym == E_TRAN) {
//                         result[i].lon() += field[k].lon() * sn;
//                         result[i].tran() += field[k].tran() * cs;
//                         result[i].vert() += field[k].vert() * sn;
//                     } else {
//                         result[i].lon() += field[k].lon() * cs;
//                         result[i].tran() += field[k].tran() * sn;
//                         result[i].vert() += field[k].vert() * cs;
//                     }
//                 }
//             }
//         }
//         return result;
//     } else {
//         if (symmetric) {
//             fft_x.execute(&(field.data()->tran()));
//             fft_yz.execute(&(field.data()->lon()));
//             fft_yz.execute(&(field.data()->vert()));
//             double dx = 0.5 * (right-left) / N;
//             RegularMesh3D src_mesh(RegularAxis(left+dx, right-dx, field.size()), RegularAxis(vpos, vpos, 1));
//             auto result = interpolate(src_mesh, field, WrappedMesh<2>(dest_mesh, SOLVER->getGeometry()),
//                                       getInterpolationMethod<INTERPOLATION_SPLINE>(field_params.method), false);
//             double L = 2. * right;
//             if (sym == E_TRAN)
//                 for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                     double x = std::fmod(dest_mesh[i][0], L);
//                     if ((-right <= x && x < 0) || x > right) { result[i].lon() = -result[i].lon(); result[i].vert() = -result[i].vert(); }
//                 }
//             else
//                 for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                     double x = std::fmod(dest_mesh[i][0], L);
//                     if ((-right <= x && x < 0) || x > right) { result[i].tran() = -result[i].tran(); }
//                 }
//             return result;
//         } else {
//             FFT::Backward1D fft(3, N, FFT::SYMMETRY_NONE);
//             fft.execute(reinterpret_cast<dcomplex*>(field.data()));
//             field[N] = field[0];
//             RegularMesh3D src_mesh(RegularAxis(left, right, field.size()), RegularAxis(vpos, vpos, 1));
//             return interpolate(src_mesh, field, WrappedMesh<2>(dest_mesh, SOLVER->getGeometry(), true),
//                                getInterpolationMethod<INTERPOLATION_SPLINE>(field_params.method), false);
//         }
//     }
}


}}} // namespace plask
