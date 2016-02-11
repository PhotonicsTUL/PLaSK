#ifndef PLASK__SOLVER_SLAB_EXPANSION_H
#define PLASK__SOLVER_SLAB_EXPANSION_H

#include <plask/plask.hpp>

#include "solver.h"
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
    SlabBase* solver;

    Expansion(SlabBase* solver): solver(solver) {}

  private:
      double glambda;
    
  protected:

    /**
     * Compute itegrals for RE and RH matrices
     * \param layer layer number
     * \param lam wavelength
     * \param glam wavelength for gain
     */
    virtual void layerIntegrals(size_t layer, double lam, double glam) = 0;
    
  public:
  
    /// Compute all expansion coefficients
    void computeIntegrals() {
        double lambda = real(2e3*M_PI/solver->k0);
        if (solver->recompute_integrals) {
            double lam;
            if (solver->lam0) {
                lam = *solver->lam0;
                glambda = (solver->always_recompute_gain)? lambda : lam;
            } else{
                lam = glambda = lambda;
            }
            size_t nlayers = lcount();
            std::exception_ptr error;
            #pragma omp parallel for
            for (size_t l = 0; l < nlayers; ++l) {
                if (error) continue;
                try {
                    layerIntegrals(l, lam, glambda);
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            solver->recompute_integrals = false;
            solver->recompute_gain_integrals = false;
        } else if (solver->recompute_gain_integrals || 
                   (solver->always_recompute_gain && !is_zero(lambda - glambda))) {
            double lam = (solver->lam0)? *solver->lam0 : lambda;
            glambda = (solver->always_recompute_gain)? lambda : lam;
            std::vector<size_t> glayers;
            size_t nlayers = lcount();
            glayers.reserve(nlayers);
            for (size_t l = 0; l != nlayers; ++l) if (solver->lgained[l]) glayers.push_back(l);
            std::exception_ptr error;
            #pragma omp parallel for
            for (size_t l = 0; l < glayers.size(); ++l) {
                if (error) continue;
                try {
                    layerIntegrals(glayers[l], lam, glambda);
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            solver->recompute_gain_integrals = false;
        }
    }

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

    /**
     * Compute vertical component of the Poynting vector for specified fields
     * \param E electric field coefficients vector
     * \param H magnetic field coefficients vector
     * \return integrated Poynting vector i.e. the total vertically emitted energy
     */
    virtual double integratePoyntingVert(const cvector& E, const cvector& H) = 0;
    
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
     * Compute electric or magnetic field on \c dst_mesh at certain level
     * \param l layer number
     * \param level destination level
     * \param E,H electric and magnetic field coefficientscients
     * \return field distribution at \c dst_mesh
     * \return field distribution at \c dst_mesh
     */
    virtual LazyData<Vec<3,dcomplex>> getField(size_t l,
                                               const shared_ptr<const typename LevelsAdapter::Level>& level,
                                               const cvector& E,
                                               const cvector& H) = 0;
};



}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_H
