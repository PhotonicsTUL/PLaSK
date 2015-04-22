#ifndef PLASK__InN_H
#define PLASK__InN_H

/** @file
This file contains undoped InN
*/

#include <plask/material/material.h>


namespace plask { namespace materials {

/**
 * Represent undoped InN, its physical properties.
 */
struct InN: public Semiconductor {

    static constexpr const char* NAME = "InN";

    std::string name() const;
    Tensor2<double> thermk(double T, double h=INFINITY) const;
    double lattC(double T, char x) const;
    double Eg(double T, double e, char point) const;
    double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const;
    Tensor2<double> Mhh(double T, double e) const;
    Tensor2<double> Mlh(double T, double e) const;

protected:
    bool isEqual(const Material& other) const;

};


}} // namespace plask::materials

#endif	//PLASK__InN_H
