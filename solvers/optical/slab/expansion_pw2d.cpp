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
   
void ExpansionPW2D::setupMaterialParameters(size_t l)
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
    
    if (!symmetric) {
        size_t N = 4 * solver->getSize() + 1;   // number of required coefficients for material parameters
        if (refine % 2 == 0) refine += 1;       // we sample at mesh points
        size_t M = refine * N;                  // number of points to sample material at
        RegularMesh1D axis0(xl, xh - (xh-xl)/M, M);
        // uśrednianie węzeł główny ± (refine-1)/2 dookoła
        
    } else {
        size_t N = 2 * solver->getSize() + 1;   // number of required coefficients for material parameters
        size_t M = refine * N;                  // number of points to sample material at
        // uśrednianie refine pomiędzy dwoma węzłami głównymi
        
        
    }
    
}

   
}}} // namespace plask
