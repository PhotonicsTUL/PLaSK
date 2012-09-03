#ifndef PLASK__AlGaN_H
#define PLASK__AlGaN_H

/** @file
This file includes undoped AlGaN
*/

#include <plask/material/material.h>
#include "GaN.h"
#include "AlN.h"

namespace plask {

/**
 * Represent undoped AlGaN, its physical properties.
 */
struct AlGaN: public Semiconductor {

    static constexpr const char* NAME = "AlGaN";

    AlGaN(const Material::Composition& Comp);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> condT(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double Eg(double T, char point) const;
    virtual double lattC(double T, char x) const;

protected:
    double Al,
           Ga;

    GaN mGaN;
    AlN mAlN;

};


} // namespace plask

#endif	//PLASK__AlGaN_H
