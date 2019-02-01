#include "expansion.h"

namespace plask { namespace optical { namespace slab {

void Expansion::getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix&, const cdiagonal&)
{
    size_t nr = Te.rows(), nc = Te.cols();
    // Eigenvector matrix is simply a unity matrix
    std::fill_n(Te.data(), nr*nc, 0.);
    std::fill_n(Te1.data(), nr*nc, 0.);
    for (size_t i = 0; i < nc; i++)
        Te(i,i) = Te1(i,i) = 1.;
}

// This is the basic relation using fields orthonormality and neglecting vertical component.
// Subclasses may override it with better formula.
double Expansion::integrateField(WhichField field, const cvector& E, const cvector& H) {
    double sum = 0.;
    if (field == FIELD_E)
        for (dcomplex e: E) sum += real(e * conj(e));
    else
        for (dcomplex h: H) sum += real(h * conj(h));
    return 0.5 * sum;
}

}}} // namespace
