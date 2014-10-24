#ifndef PLASK__InSb_Si_H
#define PLASK__InSb_Si_H

/** @file
This file contains Si-doped InSb
*/

#include <plask/material/material.h>
#include "InSb.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped InSb, its physical properties.
 */
struct InSb_Si: public InSb {

    static constexpr const char* NAME = "InSb:Si";

    InSb_Si(DopingAmountType Type, double Val);
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

};

}} // namespace plask::materials

#endif	//PLASK__InSb_Si_H
