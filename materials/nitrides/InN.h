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
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual Tensor2<double> Me(double T, char point) const;
    virtual Tensor2<double> Mhh(double T, char point) const;
    virtual Tensor2<double> Mlh(double T, char point) const;

protected:
    virtual bool isEqual(const Material& other) const;

};


} // namespace plask

#endif	//PLASK__InN_H
