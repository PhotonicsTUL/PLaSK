#ifndef PLASK__InN_H
#define PLASK__InN_H

/** @file
This file includes undoped InN
*/

#include <plask/material/material.h>


namespace plask {

/**
 * Represent undoped InN, its physical properties.
 */
struct InN: public Semiconductor {

    static constexpr const char* NAME = "InN";

    virtual std::string name() const;
    virtual std::pair<double,double> thermCond(double T) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual std::pair<double,double> Me(double T, char point) const;
    virtual std::pair<double,double> Mhh(double T, char point) const;
    virtual std::pair<double,double> Mlh(double T, char point) const;

};


} // namespace plask

#endif	//PLASK__InN_H
