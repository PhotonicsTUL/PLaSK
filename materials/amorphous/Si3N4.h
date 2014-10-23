#ifndef PLASK__Si3N4_H
#define PLASK__Si3N4_H

/** @file
This file contains Si3N4
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent Si3N4, its physical properties.
 */
struct Si3N4: public Dielectric {

    static constexpr const char* NAME = "aSi3N4";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override; //TODO
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override; //TODO
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__Si3N4_H