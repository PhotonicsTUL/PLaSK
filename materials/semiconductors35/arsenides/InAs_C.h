#ifndef PLASK__InAs_C_H
#define PLASK__InAs_C_H

/** @file
This file contains C-doped InAs
*/

#include <plask/material/material.h>
#include "InAs.h"

namespace plask { namespace materials {

/**
 * Represent C-doped InAs, its physical properties.
 */
struct InAs_C: public InAs {

    static constexpr const char* NAME = "InAs:C";

    InAs_C(DopingAmountType Type, double Val);
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
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InAs_C_H
