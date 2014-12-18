#ifndef PLASK__aSiO2_H
#define PLASK__aSiO2_H

/** @file
This file contains a-SiO2
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent a-SiO2, its physical properties.
 */
struct aSiO2: public Dielectric {

    static constexpr const char* NAME = "aSiO2";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override;
    virtual ConductivityType condtype() const override;
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__aSiO2_H
