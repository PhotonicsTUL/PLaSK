#include "expansion_pw2d.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

ExpansionPW2D::ExpansionPW2D(FourierReflection2D* solver): solver(solver)
{
    auto geometry = solver->getGeometry();

    symmetric = geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN);
    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    left = geometry->getChild()->getBoundingBox().lower[0];
    right = geometry->getChild()->getBoundingBox().upper[0];

    if (symmetric) {
        if (right < 0) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0)
            throw BadMesh(solver->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
    }

    if (!periodic) {
        // Add PMLs
        left -= solver->pml.size + solver->pml.shift;
        right += solver->pml.size + solver->pml.shift;
    }

    size_t refine = solver->refine, M;
                                                            //  . . 0 . . . . 1 . . . . 2 . . . . 0    N = 3  refine = 5  M = 15
    if (!symmetric) {                                       //  ^ ^ ^ ^ ^
        N = 2 * solver->getSize() + 1;                      // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4
        nN = 4 * solver->getSize() + 1;
        M = refine * nN;
        double dx = 0.5 * (right-left) * (1-refine) / M;    // . . 0 . . . 1 . . . 2 . .               N = 3  refine = 4  M = 12
        xmesh = RegularMesh1D(left + dx, right + dx, M);    //  ^ ^ ^ ^
    } else {                                                // |0 1 2 3|4 5 6 7|8 9 0 1
        N = solver->getSize() + 1;
        nN = 2 * solver->getSize() + 1;                     // # . 0 . # . 1 . # . 2 . # . 3 . # . 3   N = 4 refine = 4  M = 16
        M = refine * nN;                                    //  ^ ^ ^ ^
        double dx = 0.5 * (right-left) / M;                 // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5
        xmesh = RegularMesh1D(left + dx, right - dx, M);
    }

    // Compute permeability coefficients
    if (periodic) {
        mag.reset(nN, Tensor2<dcomplex>(0.));
        mag[0].c00 = 1.; mag[0].c11 = 1.; // constant 1
    } else {
        DataVector<double> Sy(M, 0.);   // PML coeffs for mu
        if (!symmetric) {
            // Add PMLs
            double pl = left + solver->pml.size, pr = right - solver->pml.size;
            pil = std::lower_bound(xmesh.begin(), xmesh.end(), pl) - xmesh.begin();
            for (size_t i = 0; i != pil; ++i) {
                double h = (pl - xmesh[i]) / solver->pml.size;
                Sy[i] = solver->pml.extinction * pow(h, solver->pml.order);
            }
            pir = std::lower_bound(xmesh.begin(), xmesh.end(), pr) - xmesh.begin();
            for (size_t i = pir+1; i != xmesh.size(); ++i) {
                double h = (xmesh[i] - pr) / solver->pml.size;
                Sy[i] = solver->pml.extinction * pow(h, solver->pml.order);
                dcomplex sy(1., Sy[i]);
            }
        } else {
            // Add PMLs
            double pr = right - solver->pml.size;
            pil = 0; pir = std::lower_bound(xmesh.begin(), xmesh.end(), pr) - xmesh.begin();
            for (size_t i = pir+1; i != xmesh.size(); ++i) {
                double h = (xmesh[i] - pr) / solver->pml.size;
                Sy[i] = solver->pml.extinction * pow(h, solver->pml.order);
                dcomplex sy(1., Sy[i]);
            }
        }
        // Average mu
        std::fill(mag.begin(), mag.end(), Tensor2<dcomplex>(0.));
        for (size_t i = 0; i != nN; ++i) {
            for (size_t j = 0; j != refine; ++j) {
                dcomplex sy(1., Sy[refine*i+j]);
                mag[i] += Tensor2<dcomplex>(sy, 1./sy);
            }
            mag[i] /= refine;
        }
        // Compute FFT
        fft.forward(2, nN, reinterpret_cast<dcomplex*>(mag.data()), symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE);
    }
}


size_t ExpansionPW2D::lcount() const {
}


bool ExpansionPW2D::diagonalQE(size_t l) const {
}


size_t ExpansionPW2D::matrixSize() const {
    return 2 * N;
}


cmatrix ExpansionPW2D::getRE(size_t l, dcomplex k0, dcomplex kx, dcomplex ky) {
}


cmatrix ExpansionPW2D::getRH(size_t l, dcomplex k0, dcomplex kx, dcomplex ky) {
}

DataVector<const Tensor3<dcomplex>> ExpansionPW2D::getMaterialParameters(size_t l)
{
    auto geometry = solver->getGeometry();
    const RectilinearMesh1D& axis1 = solver->getLayerPoints(l);

    size_t refine = solver->refine;
    size_t M = refine * nN;

    DataVector<Tensor3<dcomplex>> NR(M);

    RectilinearMesh2D mesh(xmesh, axis1, RectilinearMesh2D::ORDER_TRANSPOSED);
    auto temperature = solver->inTemperature(mesh);
    double lambda = real(solver->getWavelength());
    double maty = axis1[0]; // at each point along any vertical axis material is the same
    for (size_t i = 0; i < M; ++i) {
        auto material = geometry->getMaterial(Vec<2>(xmesh[i],maty));
        // assert([&]()->bool{for(auto y: axis1)if(geometry->getMaterial(Vec<2>(xmesh[i],y))!=material)return false; return true;}());
        double T = 0.; // average temperature in all vertical points
        for (size_t j = i * axis1.size(), end = (i+1) * axis1.size(); j != end; ++j) T += temperature[j];
        T /= axis1.size();
        NR[i] = material->NR(lambda, T);
    }

    for (Tensor3<dcomplex>& val: NR) val.sqr_inplace(); // make epsilon from NR

    // Materials coefficients (espilon and mu, which appears from PMLs)
    // Add PMLs
    if (!periodic) {
        Tensor3<dcomplex> ref;
        double pl = left + solver->pml.size, pr = right - solver->pml.size;
        ref = NR[pil];
        for (size_t i = 0; i != pil; ++i) {
            double h = (pl - xmesh[i]) / solver->pml.size;
            dcomplex sy(1., solver->pml.extinction * pow(h, solver->pml.order));
            NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
        }
        ref = NR[pir];
        for (size_t i = pir+1; i != xmesh.size(); ++i) {
            double h = (xmesh[i] - pr) / solver->pml.size;
            dcomplex sy(1., solver->pml.extinction * pow(h, solver->pml.order));
            NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
        }
    }

    // Average material parameters
    DataVector<Tensor3<dcomplex>> coeffs(nN, Tensor3<dcomplex>(0.));
    for (size_t i = 0; i != nN; ++i) {
        for (size_t j = 0; j != refine; ++j) {
            coeffs[i] += NR[refine*i + j];
        }
        coeffs[i] /= refine;
    }

    // Perform FFT
    fft.forward(5, nN, reinterpret_cast<dcomplex*>(coeffs.data()), symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE);

    // Shift coefficients to proper position 0 -> left
    if (!symmetric) {
    }

    // Cache coefficients required for field computations

    return coeffs;
}


}}} // namespace plask
