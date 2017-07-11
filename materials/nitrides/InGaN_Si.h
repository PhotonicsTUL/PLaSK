#ifndef PLASK__InGaN_Si_H
#define PLASK__InGaN_Si_H

/** @file
This file contains Si-doped InGaN
*/

#include <plask/material/material.h>
#include "InGaN.h"
#include "GaN_Si.h"
#include "InN_Si.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped InGaN, its physical properties.
 */
struct InGaN_Si: public InGaN {

    static constexpr const char* NAME = "InGaN:Si";

    InGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO change to cm^(-3)
    virtual double Na() const override;
    virtual double Nd() const override;
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

protected:
    double ND,
           Nf_RT;

    GaN_Si mGaN_Si;
    InN_Si mInN_Si;

};


}} // namespace plask::materials

#endif	//PLASK__InGaN_Si_H
