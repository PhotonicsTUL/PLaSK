#ifndef PLASK__SOLVER_SLAB_EXPANSION_H
#define PLASK__SOLVER_SLAB_EXPANSION_H

#include <plask/plask.hpp>

#include "matrices.h"
#include "meshadapter.h"

namespace plask { namespace solvers { namespace slab {

struct PLASK_SOLVER_API Expansion {

    /// Specified component in polarization or symmetry
    enum Component {
        E_UNSPECIFIED = 0,  ///< All components exist or no symmetry
        E_TRAN = 1,         ///< E_tran and H_long exist or are symmetric and E_long and H_tran anti-symmetric
        E_LONG = 2          ///< E_long and H_tran exist or are symmetric and E_tran and H_long anti-symmetric
    };

    enum WhichField {
        FIELD_E,            ///< Electric field
        FIELD_H             ///< Magnetic field
    };
    WhichField which_field;
    InterpolationMethod field_interpolation;

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
     * \param layer layer number
     * \param[out] RE,RH resulting matrix
     */
    virtual void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) = 0;

    /**
     * Get refractive index back from expansion
     * \param lay layer number
     * \param mesh mesh to get parameters to
     * \param interp interpolation method
     * \return computed refractive indices
     */
    virtual LazyData<Tensor3<dcomplex>> getMaterialNR(size_t lay,
                                                      const shared_ptr<const typename LevelsAdapter::Level> &level,
                                                      InterpolationMethod interp) = 0;
    /**
     * Prepare for computatiations of the fields
     * \param field which field is computed
     * \param method interpolation method
     */
    void initField(WhichField which, InterpolationMethod method) {
        which_field = which;
        field_interpolation = method;
        prepareField();
    }

  protected:
    /**
     * Prepare for computatiations of the fields
     */
    virtual void prepareField() {}

  public:
    /**
     * Cleanup after computatiations of the fields
     */
    virtual void cleanupField() {}

    /**
     * Compute electric og magnetic field on \c dst_mesh at certain level
     * \param l layer number
     * \param level destination level
     * \param E,H electric and magnetic field coefficientscients
     * \return field distribution at \c dst_mesh
     * \return field distribution at \c dst_mesh
     */
    virtual DataVector<const Vec<3,dcomplex>> getField(size_t l,
                                                       const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                       const cvector& E,
                                                       const cvector& H) = 0;
};



}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_H
