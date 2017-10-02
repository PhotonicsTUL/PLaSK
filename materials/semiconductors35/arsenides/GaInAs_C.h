#ifndef PLASK__GaInAs_C_H
#define PLASK__GaInAs_C_H

/** @file
This file contains C-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_C.h"
//#include "InAs_C.h"

namespace plask { namespace materials {

/**
 * Represent C-doped GaInAs, its physical properties.
 */
struct GaInAs_C: public GaInAs {

    static constexpr const char* NAME = "InGaAs:C";

    GaInAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

    //GaAs_C mGaAs_C;
    //InAs_C mInAs_C;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_C_H
