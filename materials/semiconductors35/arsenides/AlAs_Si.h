#ifndef PLASK__AlAs_Si_H
#define PLASK__AlAs_Si_H

/** @file
This file contains Si-doped AlAs
*/

#include <plask/material/material.h>
#include "AlAs.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlAs, its physical properties.
 */
struct AlAs_Si: public AlAs {

    static constexpr const char* NAME = "AlAs:Si";

    AlAs_Si(DopingAmountType Type, double Val);
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
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__AlAs_Si_H
