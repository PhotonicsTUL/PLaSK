#ifndef PLASK__InP_Be_H
#define PLASK__InP_Be_H

/** @file
This file contains Be-doped InP
*/

#include <plask/material/material.h>
#include "InP.h"

namespace plask { namespace materials {

/**
 * Represent Be-doped InP, its physical properties.
 */
struct InP_Be: public InP {

    static constexpr const char* NAME = "InP:Be";

    InP_Be(DopingAmountType Type, double Val);
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
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InP_Be_H
