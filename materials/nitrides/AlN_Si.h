#ifndef PLASK__AlN_Si_H
#define PLASK__AlN_Si_H

/** @file
This file contains Si-doped AlN
*/

#include <plask/material/material.h>
#include "AlN.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlN, its physical properties.
 */
struct AlN_Si: public AlN {

    static constexpr const char* NAME = "AlN:Si";

    AlN_Si(DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
		   mob_RT;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_Si_H
