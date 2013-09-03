#ifndef PLASK__SOLVER_SLAB_EXPANSION_H
#define PLASK__SOLVER_SLAB_EXPANSION_H

#include <plask/plask.hpp>

#include "matrices.h"

namespace plask { namespace solvers { namespace slab {

struct Expansion {

    /**
     * Return number of distinct layers
     * \return number of layers
     */
    virtual size_t lcount() const = 0;

    /**
     * Tell if matrix for i-th layer is diagonal
     * \param l layer number
     * \return \c true if the i-th matrix is diagonal
     */
    virtual bool diagonalQE(size_t l) const { return false; }

    /**
     * Return size of the expansion matrix (equal to the number of expansion coefficients)
     * \return size of the expansion matrix
     */
    virtual size_t matrixSize() const = 0;

    /**
     * Get RE matrix
     * \param l layer number
     * \param k0 normalized frequency [1/µm]
     * \param kx,ky horizontal wavevector components [1/µm]
     * \return RE matrix
     */
    virtual cmatrix getRE(size_t l, dcomplex k0, dcomplex kx, dcomplex ky) = 0;

    /**
     * Get RH matrix
     * \param l layer number
     * \param k0 normalized frequency [1/µm]
     * \param kx,ky horizontal wavevector components [1/µm]
     * \return RH matrix
     */
    virtual cmatrix getRH(size_t l, dcomplex k0, dcomplex kx, dcomplex ky) = 0;
};



}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_H
