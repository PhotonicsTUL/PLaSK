#ifndef PLASK__GaN_Mg_H
#define PLASK__GaN_Mg_H

/** @file
This file contains Mg-doped GaN
*/

#include "plask/material/material.hpp"
#include "GaN.hpp"

namespace plask { namespace materials {

/**
 * Represent Mg-doped GaN, its physical properties.
 */
struct GaN_Mg: public GaN {

    static constexpr const char* NAME = "GaN:Mg";

    GaN_Mg(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO change to cm^(-3)
    double Na() const override;
    double Nd() const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

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

    GaN_Mg_bulk(double val): GaN_Mg(val) {}

    Tensor2<double> thermk(double T, double t) const override;

};


}} // namespace plask::materials

#endif	//PLASK__GaN_Mg_H
