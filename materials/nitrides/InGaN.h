#ifndef PLASK__InGaN_H
#define PLASK__InGaN_H

/** @file
This file includes undoped InGaN
*/

#include <plask/material/material.h>
#include "GaN.h"
#include "InN.h"

namespace plask {

/**
 * Represent undoped InGaN, its physical properties.
 */
struct InGaN: public Semiconductor {

    static constexpr const char* NAME = "InGaN";

    InGaN(const Material::Composition& Comp);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual double condT(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double Eg(double T, char point) const;
    virtual double lattC(double T, char x) const;

protected:
    double In,
           Ga;

    GaN mGaN;
    InN mInN;

};


} // namespace plask

#endif	//PLASK__InGaN_H
