#ifndef PLASK__PHYS_FUNCTIONS_H
#define PLASK__PHYS_FUNCTIONS_H

#include <plask/config.h>   //for PLASK_API

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
    PLASK_API double Varshni(double Eg0K, double alpha, double beta, double T);

    /**
     * Energy of Photon
     * TODO doc
     * \param lam [nm]
     * \return [eV]
     */
    PLASK_API double PhotonEnergy(double lam);

}} // namespace plask::phys

#endif // PLASK__PHYS_FUNCTIONS_H
