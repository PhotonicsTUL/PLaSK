#ifndef PLASK__MODULE_THERMAL_NODE2D_H
#define PLASK__MODULE_THERMAL_NODE2D_H

namespace plask { namespace solvers { namespace thermal {

class Node2D // 2D node in finite element method
{
public:
    Node2D(int iNo, double iX, double iY, double iT, bool ifTConst = false); // constructor (set no, coordinates, temperature, temperature flag)

    // getters
    int getNo() const; // return node number
    double getX() const; // return x
    double getY() const; // return y
    double getT() const; // return temperature
    double getHF() const; // return heat flux
    double getConvCoeff() const; // return convection coefficient
    double getTAmb1() const; // return ambient temperature when convection is assumed
    double getEmissivity() const; // return surface emissivity
    double getTAmb2() const; // return ambient temperature when radiation is assumed
    bool ifTConst() const; // true if temperature is constant
    bool ifHFConst() const; // true if heat flux is constant
    bool ifConvection() const; // true if convection is constant
    bool ifRadiation() const; // true if radiation is assumed

    // setters
    void setT(double iT); // set temperature
    void setHF(double iHF); // set heat flux
    void setHFflag(bool ifHF); // set heat flux flag
    void setConv(double iConvCoeff, double iTAmb1); // set convection
    void setConvflag(bool ifConvection); // set convection flag
    void setRad(double iEmissivity, double iTAmb2); // set radiation
    void setRadflag(bool ifRadiation); // set radiation flag

protected:
    int mNo; // number
    double mX, mY; // coordinates
    double mT; // temperature
    double mHF; // heat flux
    double mConvCoeff; // convection coefficient
    double mTAmb1; // ambient temperature for convection boundary condition
    double mEmissivity; // surface emissivity
    double mTAmb2; // ambient temperature for radiation boundary condition
    bool mfTConst; // true if temperature is constant
    bool mfHFConst; // true if heat flux is constant
    bool mfConvection; // true if convection is assumed
    bool mfRadiation; // true if radiation is assumed
};

}}} // namespace plask::solvers::thermal
#endif
