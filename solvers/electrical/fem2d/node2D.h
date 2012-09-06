#ifndef PLASK__MODULE_ELECTRICAL_NODE2D_H
#define PLASK__MODULE_ELECTRICAL_NODE2D_H

namespace plask { namespace solvers { namespace electrical {

class Node2D // 2D node in finite element method
{
public:
    Node2D(int iNo, double iX, double iY, double iV, bool ifVConst = false); // constructor (set no, coordinates, potential, potential flag)

    // getters
    int getNo() const; // return node number
    double getX() const; // return x
    double getY() const; // return y
    double getV() const; // return potential
    bool ifVConst() const; // true if potential is constant

    // setters
    void setV(double iV); // set potential

protected:
    int mNo; // number
    double mX, mY; // coordinates
    double mV; // potential
    bool mfVConst; // true if potential is constant
};

}}} // namespace plask::solvers::electrical
#endif
