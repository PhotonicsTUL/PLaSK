#ifndef PLASK__SOLVER_SLAB_EXPANSION_H
#define PLASK__SOLVER_SLAB_EXPANSION_H

#include <plask/plask.hpp>

#include "matrices.h"

namespace plask { namespace solvers { namespace slab {

struct Expansion {

    /// Solver which performs calculations (and is the interface to the outside world)
    Solver* solver;

    Expansion(Solver* solver): solver(solver) {}

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
     * Get RE anf RH matrices
     * \param l layer number
     * \param k0 normalized frequency [1/µm]
     * \param klong,ktran horizontal wavevector components [1/µm]
     * \param[out] RE,RH resulting matrix
     */
    virtual void getMatrices(size_t l, dcomplex k0, dcomplex klong, dcomplex ktran, cmatrix& RE, cmatrix& RH) = 0;

    /**
     * Compute electric field on \c dst_mesh at certain level
     * \param l layer number
     * \param dst_mesh destination mesh
     * \param k0 normalized frequency [1/µm]
     * \param klong,ktran horizontal wavevector components [1/µm]
     * \param E,H electric and magnetic field coefficients
     * \param method interpolation method
     * \return electric field distribution at \c dst_mesh
     */
    virtual DataVector<Vec<3,dcomplex>> fieldE(size_t l, const Mesh& dst_mesh, dcomplex k0, dcomplex klong, dcomplex ktran,
                                               const cvector& E, const cvector& H, InterpolationMethod method) = 0;

    /**
     * Compute magnetic field on \c dst_mesh at certain level
     * \param l layer number
     * \param dst_mesh destination mesh
     * \param k0 normalized frequency [1/µm]
     * \param klong,ktran horizontal wavevector components [1/µm]
     * \param E,H electric and magnetic field coefficients
     * \param method interpolation method
     * \return magnetic field distribution at \c dst_mesh
     */
    virtual DataVector<Vec<3,dcomplex>> fieldH(size_t l, const Mesh& dst_mesh, dcomplex k0, dcomplex klong, dcomplex ktran,
                                               const cvector& E, const cvector& H, InterpolationMethod method) = 0;
};



}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_H
