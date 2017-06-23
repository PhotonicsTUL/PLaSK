// #ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
// #define PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
//
// #include <plask/plask.hpp>
//
// #include "expansioncyl.h"
// #include "../patterson.h"
// #include "../meshadapter.h"
//
//
// namespace plask { namespace solvers { namespace slab {
//
// struct PLASK_SOLVER_API ExpansionBesselInfini: public ExpansionBessel {
//
//     /**
//      * Create new expansion
//      * \param solver solver which performs calculations
//      */
//     ExpansionBesselInfini(BesselSolverCyl* solver);
//
//     //TODO REMOVE
//     void computeBesselZeros();
//
//     /// Perform m-specific initialization
//     void init2() override;
//
//     /// Free allocated memory
//     void reset() override;
//
//     void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;
//
// #ifndef NDEBUG
//     cmatrix epsVmm(size_t layer);
//     cmatrix epsVpp(size_t layer);
//     cmatrix epsTmm(size_t layer);
//     cmatrix epsTpp(size_t layer);
//     cmatrix epsTmp(size_t layer);
//     cmatrix epsTpm(size_t layer);
//     cmatrix epsDm(size_t layer);
//     cmatrix epsDp(size_t layer);
// #endif
// };
//
// }}} // # namespace plask::solvers::slab
//
// #endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
