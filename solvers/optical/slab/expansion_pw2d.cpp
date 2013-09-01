#include "expansion_pw2d.h"
#include "fft.h"

namespace plask { namespace solvers { namespace slab {

ExpansionPW2D::ExpansionPW2D(FourierReflection2D* solver): solver(solver)
{
    symmetric = solver->getGeometry()->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN);
}


size_t ExpansionPW2D::lcount() const {
}


bool ExpansionPW2D::diagonalQE(size_t l) const {
}


size_t ExpansionPW2D::matrixSize() const {
    if (symmetric) {
        return 2 * solver->getSize() + 2;
    } else {
        return 4 * solver->getSize() + 2;
    }
}


cmatrix ExpansionPW2D::getRE(size_t l, dcomplex k0, dcomplex kx, dcomplex ky) {
}


cmatrix ExpansionPW2D::getRH(size_t l, dcomplex k0, dcomplex kx, dcomplex ky) {
}

DataVector<const Tensor3<dcomplex>> ExpansionPW2D::getMaterialParameters(size_t l)
{
    auto geometry = solver->getGeometry();

    double xl = geometry->getChild()->getBoundingBox().lower[0];
    double xh = geometry->getChild()->getBoundingBox().upper[0];

    if (!geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN)) {
        // Add PMLs
        xl -= solver->pml.size + solver->pml.shift;
        xh += solver->pml.size + solver->pml.shift;
    }

    size_t refine = solver->refine;
    size_t N, M;    // number of of required coefficients for material parameters and ofpoints to sample material at
    const RectilinearMesh1D& axis1 = solver->getLayerPoints(l);
    RegularMesh1D axis0;

    if (!symmetric) {
        N = 4 * solver->getSize() + 1;
        if (refine % 2 == 0) ++refine;  // we sample at (refine-1)/2 secondary points per single coefficient
        M = refine * N;
        axis0 = RegularMesh1D(xl, xh - (xh-xl)/M, M);
    } else {
        N = 2 * solver->getSize() + 1;
        M = refine * N;
        double dx = 0.5 * (xh-xl) / M;
        axis0 = RegularMesh1D(xl + dx, xh - dx, M);
    }

    DataVector<Tensor3<dcomplex>> NR(M);

    RectilinearMesh2D mesh(axis0, axis1, RectilinearMesh2D::ORDER_TRANSPOSED);
    auto temperature = solver->inTemperature(mesh);
    double lambda = real(solver->getWavelength());
    double maty = axis1[0]; // at each point along any vertical axis material is the same
    for (size_t i = 0; i < M; ++i) {
        auto material = geometry->getMaterial(Vec<2>(axis0[i],maty));
        // assert([&]()->bool{for(auto y: axis1)if(geometry->getMaterial(Vec<2>(axis0[i],y))!=material)return false; return true;}());
        double T = 0.; // average temperature in all vertical points
        for (size_t j = i * axis1.size(), end = (i+1) * axis1.size(); j != end; ++j) T += temperature[j];
        T /= axis1.size();
        NR[i] = material->NR(lambda, T);
    }

    // Materials coefficients (espilon and mu, which appears from PMLs)
    DataVector<Tensor3<dcomplex>> coeffs(2*N, Tensor3<dcomplex>(1.));

    if (!symmetric) {
        // Add PMLs
        // Average material parameters (uśrednianie węzeł główny ± (refine-1)/2 dookoła)
        // Perform FFT
    } else {
        // Add PMLs
        // Average material parameters (uśrednianie refine pomiędzy dwoma węzłami głównymi)
        // Perform FFT
    }

    // Cache coefficients required for field computations

    return coeffs;
}


}}} // namespace plask
