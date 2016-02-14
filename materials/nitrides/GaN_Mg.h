#ifndef PLASK__GaN_Mg_H
#define PLASK__GaN_Mg_H

/** @file
This file contains Mg-doped GaN
*/

#include <plask/material/material.h>
#include "GaN.h"

namespace plask { namespace materials {

/**
 * Represent Mg-doped GaN, its physical properties.
 */
struct GaN_Mg: public GaN {

    static constexpr const char* NAME = "GaN:Mg";

    GaN_Mg(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
	virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Na() const override;
    virtual double Nd() const override;
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double NA,
           Nf_RT,
		   mob_RT,
		   cond_RT;

};


/**
 * Represent Mg-doped bulk (substrate) GaN, its physical properties.
 */
struct GaN_Mg_bulk: public GaN_Mg {

    static constexpr const char* NAME = "GaN_bulk:Mg";

    GaN_Mg_bulk(DopingAmountType type, double val): GaN_Mg(type, val) {}

    virtual Tensor2<double> thermk(double T, double t) const override;

};


}} // namespace plask::materials

#endif	//PLASK__GaN_Mg_H
