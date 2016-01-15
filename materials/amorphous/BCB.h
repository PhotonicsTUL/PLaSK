#ifndef PLASK__BCB_H
#define PLASK__BCB_H

/** @file
This file contains BCB (benzo-cyclo-butene)
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent BCB (benzo-cyclo-butene), its physical properties.
 */
struct BCB: public Dielectric {

    static constexpr const char* NAME = "BCB";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override;
    virtual ConductivityType condtype() const override;
    virtual double dens(double T) const override;
    virtual double cp(double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__BCB_H
