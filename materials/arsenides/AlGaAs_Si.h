#ifndef PLASK__AlGaAs_Si_H
#define PLASK__AlGaAs_Si_H

/** @file
This file includes Si-doped AlGaAs
*/

#include <plask/material/material.h>
#include "AlGaAs.h"
#include "GaAs_Si.h"
#include "AlAs_Si.h"

namespace plask {

/**
 * Represent Si-doped AlGaAs, its physical properties.
 */
struct AlGaAs_Si: public AlGaAs {

    static constexpr const char* NAME = "AlGaAs:Si";

    AlGaAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

    GaAs_Si mGaAs_Si;
    AlAs_Si mAlAs_Si;
};

} // namespace plask

#endif	//PLASK__AlGaAs_Si_H
