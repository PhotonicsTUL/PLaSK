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
#include "AlAs.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PROPERTY(AlAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double AlAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.6611 + 2.90e-5 * (T-300.);
    return tLattC;
}

MI_PROPERTY(AlAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double AlAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.099, 0.885e-3, 530., T);
    else if (point == 'X') tEg = phys::Varshni(2.24, 0.70e-3, 530., T);
    else if (point == 'L') tEg = phys::Varshni(2.46, 0.605e-3, 204., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(3.099, 0.885e-3, 530., T);
        double tEgX = phys::Varshni(2.24, 0.70e-3, 530., T);
        double tEgL = phys::Varshni(2.46, 0.605e-3, 204., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::Dso(double /*T*/, double /*e*/) const {
    return 0.28;
}

MI_PROPERTY(AlAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MINote("no temperature dependence")
           )
Tensor2<double> AlAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.124), tMeX(0.71), tMeL(0.78);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == 'L') {
        tMe.c00 = tMeL; tMe.c11 = tMeL;
    }
    else if (point == '*') {
        double EgG = Eg(T,e,'G'), EgX = Eg(T,e,'X'), EgL = Eg(T,e,'L');
        if (EgG <= EgX && EgG <= EgL) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        } else if (EgX <= EgL){
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        } else {
            tMe.c00 = tMeL; tMe.c11 = tMeL;
        }
    } else
        throw Exception("AlAs: Me: bad point '{c}'", point);
    return ( tMe );
}

MI_PROPERTY(AlAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
           )
Tensor2<double> AlAs::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.51, 0.51);
}

MI_PROPERTY(AlAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence")
           )
Tensor2<double> AlAs::Mlh(double /*T*/, double /*e*/) const {
    Tensor2<double> tMlh(0.18, 0.18);
    return tMlh;
}

MI_PROPERTY(AlAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MINote("no temperature dependence"),
            MINote("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlAs, y1,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7"),
	MINote("no temperature dependence")
)
double AlAs::y1() const {
	return 3.76;
}

MI_PROPERTY(AlAs, y2,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7"),
	MINote("no temperature dependence")
)
double AlAs::y2() const {
	return 0.90;
}

MI_PROPERTY(AlAs, y3,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7"),
	MINote("no temperature dependence")
)
double AlAs::y3() const {
	return 1.42;
}

MI_PROPERTY(AlAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double AlAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point, 'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}

MI_PROPERTY(AlAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-1.33);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(AlAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::ac(double /*T*/) const {
    return -5.64;
}

MI_PROPERTY(AlAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::av(double /*T*/) const {
    return 2.47;
}

MI_PROPERTY(AlAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::b(double /*T*/) const {
    return -2.3;
}

MI_PROPERTY(AlAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::d(double /*T*/) const {
    return -3.4;
}

MI_PROPERTY(AlAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::c11(double /*T*/) const {
    return 125.0;
}

MI_PROPERTY(AlAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::c12(double /*T*/) const {
    return 53.4;
}

MI_PROPERTY(AlAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MINote("no temperature dependence")
           )
double AlAs::c44(double /*T*/) const {
    return 54.2;
}

MI_PROPERTY(AlAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67"), // 300 K
            MISource("S. Adachi, Properties of Aluminium Gallium Arsenide, INSPEC (1993) p.48") // temperature dependence
           )
Tensor2<double> AlAs::thermk(double T, double /*t*/) const {
    double tk = 91. * pow((300./T),1.375);
    return(Tensor2<double>(tk, tk));
}

MI_PROPERTY(AlAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MINote("no temperature dependence")
            )
double AlAs::dens(double /*T*/) const { return 3.73016e3; }

MI_PROPERTY(AlAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MINote("no temperature dependence")
            )
double AlAs::cp(double /*T*/) const { return 0.424e3; }

Material::ConductivityType AlAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlAs, nr,
            //MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            //MINote("fit by Lukasz Piskorski")
            MISource("S. Gehrsitz, J. Appl. Phys. 87 (2000) 7825-7837"),
            MINote("fit by Leszek Frasunkiewicz")
           )
double AlAs::nr(double lam, double T, double /*n*/) const {
    double Al = 1.;
    double A = -0.4490233379*Al + 3.25759049;
    double B = 29.22871618*pow(Al,(-2.35349122*pow(Al,8.844978824)));
    double C = -304.7269552*Al*Al*Al + 335.1918592*Al*Al + 194.6726344*Al - 559.6098207;

    //dn/dT
    double Ad = 0.0003442176581*Al*Al*Al - 0.0005412098145*Al*Al + 0.00008671640556*Al + 0.0002093262406;
    double Bd = 132.1382231*exp(-8.32822628*Al*Al*Al + 14.65496754*Al*Al - 7.135900438*Al);
    double Cd = 117.24*Al -689.06;
    double dndT = Ad*exp(Bd / (lam + Cd));

    return ( A*exp(B / (lam + C)) + dndT*(T-296) );
    //old
    //double L2 = lam*lam*1e-6;
    //double nR296K = sqrt(1.+7.055*L2/(L2-0.068));
    //return ( nR296K + nR296K*4.6e-5*(T-296.) );
}

MI_PROPERTY(AlAs, absp,
            MISource(""),
            MINote("calculated as for Si-doped AlAs but with n = 1e16")
           )
double AlAs::absp(double lam, double T) const {
    double tEgRef300 = phys::Varshni(1.519, 0.5405e-3, 204., T);
    double tEgT = Eg(T,0.,'X');
    double tDWl = phys::h_eVc1e9*(tEgRef300-tEgT)/(tEgRef300*tEgT);
    double tWl = (lam-tDWl)*1e-3;
    double tAbsp(0.);
    double tN = 1e16; // concentration for undoped GaAs
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (tN/1e18)*(1e24*exp(-tWl/0.0169)+4.67+0.00211*pow(tWl,4.80));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (tN/1e18)*(-8.4+0.233*pow(tWl,2.6));
    return ( tAbsp );
    //return 0.;
}

MI_PROPERTY(AlAs, eps,
            MISource("http://www.iue.tuwien.ac.at/phd/quay/node27.html")
           )
double AlAs::eps(double /*T*/) const {
    return 10.1;
}

std::string AlAs::name() const { return NAME; }

bool AlAs::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<AlAs> materialDB_register_AlAs;

}}       // namespace plask::materials
