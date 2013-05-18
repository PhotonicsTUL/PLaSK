#ifndef PLASK__InAs_Si_H
#define PLASK__InAs_Si_H

/** @file
This file contains Si-doped InAs
*/

#include <plask/material/material.h>
#include "InAs.h"

namespace plask {

/**
 * Represent Si-doped InAs, its physical properties.
 */
struct InAs_Si: public InAs {

    static constexpr const char* NAME = "InAs:Si";

    InAs_Si(DopingAmountType Type, double Val);
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

} // namespace plask

#endif	//PLASK__InAs_Si_H
