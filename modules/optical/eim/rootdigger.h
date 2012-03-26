/***************************************************************************
 *  File:   rootdigger.h
 *  Descr:  Class searching roots of char_val
 *  Author: Maciej Dems <maciej.dems@p.lodz.pl>
 *  Id:     $Id: rootdigger.h 405 2012-03-08 10:44:14Z maciek $
 ***************************************************************************/

/***************************************************************************
 *   pslab - A photonic slab simulation tool                               *
 *   Copyright (C) 2005-2011 by Maciej Dems <maciej.dems@p.lodz.pl>        *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

//**************************************************************************
#ifndef ROOTDIGGER_H
#define ROOTDIGGER_H

//**************************************************************************
#include <vector>
#include <limits>
#include <string>
/*
//**************************************************************************
class RootDigger {
  protected:
    // The solver providing char_val
    RootSolverBase& solver;
    friend class RootSolverBase;

    // Parameters for Broyden algorithm
    const double alpha, lambda_min, EPS;

    // Return Jacobian of F(x)
    void fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji) const;

    // Search for the new point x along direction p for which f decreased sufficiently
    bool lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax) const;

    // Search for the root of char_val using globally convergent Broyden method
    dcomplex Broyden(dcomplex x) const;

    // Look for map browsing through given points
    std::vector<dcomplex> find_map(std::vector<double> repoints, std::vector<double> impoints) const;

  public:
    // Parameters for Broyden algorithm
    double tolx, tolf_min, tolf_max, maxstep;

    // Maximum number of iterations
    int maxiterations;

    // Constructors
    RootDigger(RootSolverBase& solv) : solver(solv),
        maxiterations(500),                              // maximum number of iterations
        tolx(1.0e-11),                              // absolute tolerance on the argument
        tolf_min(1.0e-15),                          // sufficient tolerance on the function value
        tolf_max(1.0e-12),                          // required tolerance on the function value
        maxstep(0.1),                              // maximum step in one iteration
        alpha(1.0e-4),                              // ensures sufficient decrease of charval in each step
        lambda_min(1.0e-7),                         // minimum decresed ratio of the step (lambda)
        EPS(sqrt(std::numeric_limits<double>::epsilon()))// square root of machine precission
    {};

    RootDigger(RootDigger& d) : solver(d.solver), maxiterations(d.maxiterations), tolx(d.tolx),
                                tolf_min(d.tolf_min), tolf_max(d.tolf_max), alpha(d.alpha),
                                maxstep(d.maxstep), lambda_min(d.lambda_min), EPS(d.EPS) {};

    /// Search for single mode within the region real(start) - real(end),
    // imag(start) - imag(end), divided on: replot for real direction
    // implot - for imaginary one
    // return complex coordinate of the mode
    // return 0 if the mode shas not been found
    inline dcomplex searchMode(dcomplex start,dcomplex end, int replot,int implot) {
        std::vector<dcomplex> vec;
        vec = searchModes(start, end, replot, implot, 1);
        if (vec.size() == 0)
            throw "RootDigger::SearchMode: did not find any mode";
        else
            return vec[0];
    };

    /// Search for modes within the region real(start) - real(end),
    //   imag(start) - imag(end), divided on: replot for real direction and implot for imaginary one
    //   search for single mode defined by the number: modeno and allmodes==false
    //   search for the given number of modes: number of the searched modes is modeno and allmodes==true
    //   return vector of complex coordinates of the modes
    //   return 0 size vector if the mode has not been found
    std::vector<dcomplex> searchModes(dcomplex start, dcomplex end, int replot, int implot, int num_modes);

    /// Search for a single mode starting from the given point: point
    // return complex coordinate of the mode
    // return 0 if the mode has not been found
    dcomplex getMode(dcomplex point) const;
};
*/
//**************************************************************************
#endif // ROTDIGGER_H
