#ifndef PLASK__GaAs_Be_H
#define PLASK__GaAs_Be_H

/** @file
This file contains Be-doped GaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"

namespace plask { namespace materials {

/**
 * Represent Be-doped GaAs, its physical properties.
 */
struct GaAs_Be: public GaAs {

    static constexpr const char* NAME = "GaAs:Be";

    GaAs_Be(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaAs_Be_H
