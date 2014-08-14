#ifndef PLASK__SOLVER_SLAB_REFLECTION_H
#define PLASK__SOLVER_SLAB_REFLECTION_H

#include "matrices.h"
#include "transfer.h"
#include "solver.h"


namespace plask { namespace solvers { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API ReflectionTransfer: public Transfer {

    /// Struct containing data for computing field in a layer
    struct LayerFields {
        cvector F, B;
    };

  protected:

    cmatrix P;                                  ///< Current reflection matrix
    bool allP;                                  ///< Do we need to keep all the P matrices?

    std::vector<LayerFields> fields;            ///< Vector of fields computed for each layer

  private:

    cdiagonal phas;                             ///< Current phase shift matrix
    int* ipiv;                                  ///< Pivot vector
    std::vector<cmatrix> memP;                  ///< Reflection matrices for each layer

  public:

    ReflectionTransfer(SlabBase* solver, Expansion& expansion);

    ~ReflectionTransfer();

    cvector getReflectionVector(const cvector& incident, IncidentDirection direction) override;

    cvector getTransmissionVector(const cvector& incident, IncidentDirection side) override;

  protected:

    void getFinalMatrix() override {
        getAM(0, solver->interface-1, false);
        getAM(solver->stack.size()-1, solver->interface, true);
    }

    void determineFields() override;

    void determineReflectedFields(const cvector& incident, IncidentDirection side) override;

    cvector getFieldVectorE(double z, int n) override;

    cvector getFieldVectorH(double z, int n) override;

    /**
     * Get admittance (A) and discontinuity (M) matrices for half of the structure
     * \param start start of the transfer
     * \param end end of the transfer
     * \param add if \c true then M matrix is added to the previous value
     * \param mfac factor to multiply M matrix befere addition
     */
    void getAM(size_t start, size_t end, bool add, double mfac=1.);

    /**
     * Find reflection matrix for the part of the structure
     * \param start starting layer
     * \param end last layer (reflection matrix is computed for this layer)
     * \param emitting should the reflection matrix in the first layer be 0?
     */
    void findReflection(int start, int end, bool emitting);

    /**
     * Store P matrix if we want it for field computation
     * \param n layer number
     */
    void storeP(size_t n);
};


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_REFLECTION_H
