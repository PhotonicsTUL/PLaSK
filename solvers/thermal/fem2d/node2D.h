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
    bool ifTConst() const; // true if temperature is constant

    // setters
    void setT(double iT); // set temperature

protected:
    int mNo; // number
    double mX, mY; // coordinates
    double mT; // temperature
    bool mfTConst; // true if temperature is constant
};

}}} // namespace plask::solvers::thermal
#endif
