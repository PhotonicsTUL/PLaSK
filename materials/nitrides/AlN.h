#ifndef PLASK__AlN_H
#define PLASK__AlN_H

/** @file
This file includes undoped AlN
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped AlN, its physical properties.
 */
struct AlN: public Semiconductor {

    static constexpr const char* NAME = "AlN";

	virtual std::string name() const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual std::pair<double,double> Me(double T, char point) const;
/*TODO
    virtual double Mhh(double T, char point) const;
    virtual double Mhh_l(double T, char point) const;
    virtual double Mhh_v(double T, char point) const;
    virtual double Mlh(double T, char point) const;
    virtual double Mlh_l(double T, char point) const;
    virtual double Mlh_v(double T, char point) const;
*/
};


} // namespace plask

#endif	//PLASK__AlN_H
