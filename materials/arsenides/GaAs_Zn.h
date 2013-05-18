#ifndef PLASK__GaAs_Zn_H
#define PLASK__GaAs_Zn_H

/** @file
This file contains Zn-doped GaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"

namespace plask {

/**
 * Represent Zn-doped GaAs, its physical properties.
 */
struct GaAs_Zn: public GaAs {

    static constexpr const char* NAME = "GaAs:Zn";

    GaAs_Zn(DopingAmountType Type, double Val);
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

} // namespace plask

#endif	//PLASK__GaAs_Zn_H
