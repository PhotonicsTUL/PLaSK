#ifndef PLASK__PHYS_FUNCTIONS_H
#define PLASK__PHYS_FUNCTIONS_H

namespace plask { namespace phys {

    /**
     * Varshni functions
     * TODO doc
     * \param Eg0K [eV]
     * \param alpha [eV/K]
     * \param beta [K]
     * \param T [K]
     * \return [eV]
     */
    double Varshni(double Eg0K, double alpha, double beta, double T);

}} // namespace plask::phys

#endif // PLASK__PHYS_FUNCTIONS_H
