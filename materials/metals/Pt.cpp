/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "Pt.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Pt::name() const { return NAME; }

MI_PROPERTY(Pt, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MINote("no temperature dependence")
)

MI_PROPERTY(Pt, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MINote("no temperature dependence")
)

MI_PROPERTY(Pt, Nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MINote("no temperature dependence")
)

Pt::Pt(): LorentzDrudeMetal(9.59,
                            {0.333,  0.191,  0.659,  0.547,  3.576}, // f
                            {0.080,  0.517,  1.838,  3.668,  8.517}, // G
                            {0.000,  0.780,  1.314,  3.141,  9.249}  // w
) {}

// Pt::Pt(): BrendelBormannMetal(9.59,
//                               {0.333,  0.186,  0.665,  0.551,  2.214}, // f
//                               {0.080,  0.498,  1.851,  2.604,  2.891}, // G
//                               {0.000,  0.782,  1.317,  3.189,  8.236}, // w
//                               {0.000,  0.031,  0.096,  0.766,  1.146}  // s
// ) {}


MI_PROPERTY(Pt, cond,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, pp. 2121-2122, 2005."),
            MINote("fit from: ?Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Pt::cond(double T) const {
    double tCond = 1. / (3.84e-10*(T-300.)+1.071e-7);
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(Pt, thermk,
            MISource("CRC Handbook of Chemistry and Physics, Internet Version 2005, http://www.hbcpnetbase.com, edited by D.R. Lide, CRC Press, Boca Raton, FL, sec. 12, 2005."),
            MINote("fit from: Lukasz Piskorski, PhD thesis, 2010"),
            MIArgumentRange(MaterialInfo::T, 200, 500)
            )
Tensor2<double> Pt::thermk(double T, double /*t*/) const {
    const double tCondT = 3.6e-5*pow(T-300.,2.) - 4e-3*(T-300.) + 71.7;
    return ( Tensor2<double>(tCondT, tCondT) );
}

//MI_PROPERTY(Pt, absp,
//            MISource(""),
//            MINote("TODO"),
//            MIArgumentRange(MaterialInfo::lam, 400, 12400)
//            )
//double Pt::absp(double lam, double /*T*/) const {
//    const double ulam = lam*1e-3;
//    if (ulam<2500.)
//        return ( 1e6*(2.80215 - 15.3234*ulam + 51.5342*ulam*ulam -94.3547*pow(ulam,3.) + 101.1011*pow(ulam,4.) -65.11963*pow(ulam,5.) + 24.741*pow(ulam,6.) - 5.099038*pow(ulam,7.) + 0.4391658*pow(ulam,8.)) );
//    else
//        return ( -39538.4 + 305946*ulam - 67838.1*ulam*ulam + 7492.84*pow(ulam,3.) - 417.401*pow(ulam,4.) + 9.27859*pow(ulam,5.) );
//}

bool Pt::isEqual(const Material &/*other*/) const {
    return true;
}

//MI_PROPERTY(Pt, nr,
//            MISource(""),
//            MINote("TODO"),
//            MIArgumentRange(MaterialInfo::lam, 280, 12400)
//            )
//double Pt::nr(double lam, double /*T*/, double /*n*/) const {
//    const double ulam = lam*1e-3;
//    if (ulam<3700.)
//        return ( 2.20873*exp(-2.70386*pow(ulam-1.76515,2.)) + 0.438205 + 3.87609*ulam - 1.5836*ulam*ulam+0.197125*pow(ulam,3.) );
//    else
//        return ( 3.43266 - 0.963058*ulam + 0.260552*ulam*ulam - 0.00791393*pow(ulam,3.) );
//}

static MaterialsDB::Register<Pt> materialDB_register_Pt;

}}       // namespace plask::materials
