#ifndef PLASK__SOLVER_SLAB_EXPANSION_H
#define PLASK__SOLVER_SLAB_EXPANSION_H

#include <plask/plask.hpp>

#include "solver.h"
#include "matrices.h"
#include "meshadapter.h"

namespace plask { namespace optical { namespace slab {

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

    /// Frequency for which the actual computations are performed
    dcomplex k0;

    /// Material parameters wavelength
    double lam0;

    Expansion(SlabBase* solver): solver(solver), k0(NAN), lam0(NAN) {}

  private:
      double glambda;

  protected:

    /**
     * Method called before layer integrals are computed
     */
    virtual void prepareIntegrals(double PLASK_UNUSED(lam), double PLASK_UNUSED(glam)) {}

    /**
     * Method called after layer integrals are computed
     */
    virtual void cleanupIntegrals(double PLASK_UNUSED(lam), double PLASK_UNUSED(glam)) {}

    /**
     * Compute itegrals for RE and RH matrices
     * \param layer layer number
     * \param lam wavelength
     * \param glam wavelength for gain
     */
    virtual void layerIntegrals(size_t layer, double lam, double glam) = 0;

  public:

    /// Get lam0
    double getLam0() const { return lam0; }
    /// Set lam0
    void setLam0(double lam) {
        if (lam != lam0 && !(isnan(lam0) && isnan(lam))) {
            lam0 = lam;
            solver->recompute_integrals = true;
            solver->clearFields();
        }
    }
    /// Clear lam0
    void clearLam0() {
        if (!isnan(lam0)) {
            lam0 = NAN;
            solver->recompute_integrals = true;
            solver->clearFields();
        }
    }

    /// Get current k0
    dcomplex getK0() const { return k0; }
    /// Set current k0
    void setK0(dcomplex k) {
        if (k != k0) {
            k0 = k;
            if (k0 == 0.) k0 = 1e-12;
            if (isnan(lam0)) solver->recompute_integrals = true;
            solver->clearFields();
        }
    }

    /// Compute all expansion coefficients
    void computeIntegrals() {
        double lambda = real(2e3*PI/k0);
        if (solver->recompute_integrals) {
            double lam;
            if (!isnan(lam0)) {
                lam = lam0;
                glambda = (solver->always_recompute_gain)? lambda : lam;
            } else{
                lam = glambda = lambda;
            }
            size_t nlayers = solver->lcount;
            std::exception_ptr error;
            prepareIntegrals(lam, glambda);
            #pragma omp parallel for
            for (plask::openmp_size_t l = 0; l < nlayers; ++l) {
                if (error) continue;
                try {
                    layerIntegrals(l, lam, glambda);
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            cleanupIntegrals(lam, glambda);
            if (error) std::rethrow_exception(error);
            solver->recompute_integrals = false;
            solver->recompute_gain_integrals = false;
        } else if (solver->recompute_gain_integrals ||
                   (solver->always_recompute_gain && !is_zero(lambda - glambda))) {
            double lam = isnan(lam0)? lambda : solver->lam0;
            glambda = (solver->always_recompute_gain)? lambda : lam;
            std::vector<size_t> glayers;
            size_t nlayers = solver->lcount;
            glayers.reserve(nlayers);
            for (size_t l = 0; l != nlayers; ++l) if (solver->lgained[l]) glayers.push_back(l);
            std::exception_ptr error;
            prepareIntegrals(lam, glambda);
            #pragma omp parallel for
            for (plask::openmp_size_t l = 0; l < glayers.size(); ++l) {
                if (error) continue;
                try {
                    layerIntegrals(glayers[l], lam, glambda);
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            cleanupIntegrals(lam, glambda);
            if (error) std::rethrow_exception(error);
            solver->recompute_gain_integrals = false;
        }
    }

    /**
     * Tell if matrix for i-th layer is diagonal
     * \param l layer number
     * \return \c true if the i-th matrix is diagonal
     */
    virtual bool diagonalQE(size_t PLASK_UNUSED(l)) const { return false; }

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

    /**
     * Compute vertical component of the Poynting vector for fields specified as matrix column
     * \param n column number
     * \param TE electric field coefficients matrix
     * \param TH magnetic field coefficients matrix
     * \return integrated Poynting vector i.e. the total vertically emitted energy
     */
    inline double getModeFlux(size_t n, const cmatrix& TE, const cmatrix& TH) {
        const cvector E(const_cast<dcomplex*>(TE.data()) + n*TE.rows(), TE.rows()),
                      H(const_cast<dcomplex*>(TH.data()) + n*TH.rows(), TH.rows());
        return integratePoyntingVert(E, H);
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
