#ifndef PLASK__InN_Mg_H
#define PLASK__InN_Mg_H

/** @file
This file includes Mg-doped InN
*/

#include <plask/material/material.h>
#include "InN.h"

namespace plask {

/**
 * Represent Mg-doped InN, its physical properties.
 */
struct InN_Mg: public InN {

    static constexpr const char* NAME = "InN:Mg";

    InN_Mg(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
	virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


} // namespace plask

#endif	//PLASK__InN_Mg_H
