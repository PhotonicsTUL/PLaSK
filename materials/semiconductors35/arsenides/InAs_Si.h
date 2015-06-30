#ifndef PLASK__InAs_Si_H
#define PLASK__InAs_Si_H

/** @file
This file contains Si-doped InAs
*/

#include <plask/material/material.h>
#include "InAs.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped InAs, its physical properties.
 */
struct InAs_Si: public InAs {

    static constexpr const char* NAME = "InAs:Si";

    InAs_Si(DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InAs_Si_H
