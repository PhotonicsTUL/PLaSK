#ifndef PLASK__AlGaN_Si_H
#define PLASK__AlGaN_Si_H

/** @file
This file contains Si-doped AlGaN
*/

#include <plask/material/material.h>
#include "AlGaN.h"
#include "GaN_Si.h"
#include "AlN_Si.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlGaN, its physical properties.
 */
struct AlGaN_Si: public AlGaN {

    static constexpr const char* NAME = "AlGaN:Si";

    AlGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT;

    GaN_Si mGaN_Si;
    AlN_Si mAlN_Si;
};

}} // namespace plask::materials

#endif	//PLASK__AlGaN_Si_H
