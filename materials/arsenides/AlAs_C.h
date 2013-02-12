#ifndef PLASK__AlAs_C_H
#define PLASK__AlAs_C_H

/** @file
This file includes C-doped AlAs
*/

#include <plask/material/material.h>
#include "AlAs.h"

namespace plask {

/**
 * Represent C-doped AlAs, its physical properties.
 */
struct AlAs_C: public AlAs {

    static constexpr const char* NAME = "AlAs:C";

    AlAs_C(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

} // namespace plask

#endif	//PLASK__AlAs_C_H
