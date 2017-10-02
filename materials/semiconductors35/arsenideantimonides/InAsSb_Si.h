#ifndef PLASK__InAsSb_Te_H
#define PLASK__InAsSb_Te_H

/** @file
This file contains Si-doped InAsSb
*/

#include <plask/material/material.h>
#include "InAsSb.h"
//#include "InAs_Si.h"
//#include "InSb_Si.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped InAsSb, its physical properties.
 */
struct InAsSb_Si: public InAsSb {

    static constexpr const char* NAME = "InAsSb:Si";

    InAsSb_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double nr(double lam, double T, double n = .0) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //InAs_Si mInAs_Si;
    //InSb_Si mInSb_Si;
};

}} // namespace plask::materials

#endif	//PLASK__InAsSb_Si_H
