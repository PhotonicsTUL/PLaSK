#ifndef PLASK__InP_Si_H
#define PLASK__InP_Si_H

/** @file
This file contains Si-doped InP
*/

#include <plask/material/material.h>
#include "InP.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped InP, its physical properties.
 */
struct InP_Si: public InP {

    static constexpr const char* NAME = "InP:Si";

    InP_Si(DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InP_Si_H
