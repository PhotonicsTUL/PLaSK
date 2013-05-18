#ifndef PLASK__GaInAs_Si_H
#define PLASK__GaInAs_Si_H

/** @file
This file contains Si-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Si.h"
//#include "InAs_Si.h"

namespace plask {

/**
 * Represent Si-doped GaInAs, its physical properties.
 */
struct GaInAs_Si: public GaInAs {

    static constexpr const char* NAME = "GaInAs:Si";

    GaInAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //GaAs_Si mGaAs_Si;
    //InAs_Si mInAs_Si;
};

} // namespace plask

#endif	//PLASK__GaInAs_Si_H
