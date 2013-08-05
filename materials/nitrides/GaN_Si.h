#ifndef PLASK__GaN_Si_H
#define PLASK__GaN_Si_H

/** @file
This file contains Si-doped GaN
*/

#include <plask/material/material.h>
#include "GaN.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped GaN, its physical properties.
 */
struct GaN_Si: public GaN {

    static constexpr const char* NAME = "GaN:Si";

    GaN_Si(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
	virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

};


}} // namespace plask::materials

#endif	//PLASK__GaN_Si_H
