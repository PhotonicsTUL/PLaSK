#ifndef PLASK__AlAs_C_H
#define PLASK__AlAs_C_H

/** @file
This file contains C-doped AlAs
*/

#include <plask/material/material.h>
#include "AlAs.h"

namespace plask { namespace materials {

/**
 * Represent C-doped AlAs, its physical properties.
 */
struct AlAs_C: public AlAs {

    static constexpr const char* NAME = "AlAs:C";

    AlAs_C(DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__AlAs_C_H
