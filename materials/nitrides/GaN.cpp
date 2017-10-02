#include "GaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaN::name() const { return NAME; }

MI_PROPERTY(GaN, cond,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("G. Koblmuller et al., Appl. Phys. Lett. 91 (2007) 221905"),
            MIArgumentRange(MaterialInfo::T, 270, 400)
            )
Tensor2<double> GaN::cond(double T) const {
    double tCond = 255. * std::pow((T/300.),-0.18);
    return Tensor2<double>(tCond,tCond);
}

MI_PROPERTY(GaN, thermk,
            MISource("C. Mion et al., App. Phys. Lett. 89 (2006) 092123"),
            MIArgumentRange(MaterialInfo::T, 300, 450)
            )
Tensor2<double> GaN::thermk(double T, double t) const {
    double fun_t = std::pow((tanh(0.001529*pow(t,0.984))),0.12),
           tCondT = 230. * pow((T/300.),-1.43);
    // return Tensor2<double>(tCondT, fun_t * tCondT);
    tCondT *= fun_t;
    return Tensor2<double>(tCondT, tCondT);
 }

MI_PROPERTY(GaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double GaN::absp(double lam, double T) const {
    double dE = phys::h_eVc1e9/lam - Eg(T, 0., 'G');
    return 19000.*exp(dE/0.019) + 330*exp(dE/0.07);
}

MI_PROPERTY(GaN, nr,
            MISource("fit - Maciej Kuc"),
            MIArgumentRange(MaterialInfo::lam, 300, 580),
            MIComment("no temperature dependence")
            )
double GaN::nr(double lam, double T, double n) const {
    double tE = phys::h_eVc1e9 / lam - (Eg(T) - Eg(300.)), nR;

    if (1.000 < tE && tE <= 2.138) nR = 0.013914*tE*tE*tE*tE - 0.096422*tE*tE*tE + 0.27318*tE*tE - 0.27725*tE + 2.3535;  // lambda: 580nm - 1240nm
    else if (tE <= 3.163) nR = 0.1152*tE*tE*tE - 0.7955*tE*tE + 1.959*tE + 0.68;                                         // lambda: 392nm - 580nm
    else if (tE <= 3.351) nR = 18.2292*tE*tE*tE - 174.6974*tE*tE + 558.535*tE - 593.164;                                 // lambda: 370nm - 392nm
    else if (tE <= 3.532) nR = 33.63905*tE*tE*tE - 353.1446*tE*tE + 1235.0168*tE - 1436.09;                              // lambda: 351nm - 370nm
    else if (tE <= 4.100) nR = -0.72116*tE*tE*tE + 8.8092*tE*tE - 35.8878*tE + 51.335;                                   // lambda: 336nm - 351nm
    else if (tE <= 5.000) nR = 0.351664*tE*tE*tE*tE - 6.06337*tE*tE*tE + 39.2317*tE*tE - 112.865*tE + 124.358;           // lambda: 248nm - 336nm
    else nR = NAN;

    return nR;
}

MI_PROPERTY(GaN, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double GaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 3.1896;
    else if (x == 'c') tLattC = 5.1855;
    return tLattC;
}

MI_PROPERTY(GaN, Eg,
            MISource("Vurgaftman et al. in Piprek 2007 Nitride Semicondcuctor Devices")
            )
double GaN::Eg(double T, double e, char point) const {
    if (point == 'G' || point == '*') return phys::Varshni(3.510, 0.914e-3, 825., T);
    else return NAN;
}

/*double GaN::VB(double T, double e, char point, char hole) const {
    return 0.;
}*/

double GaN::Dso(double T, double e) const {
    return 0.017;
}

MI_PROPERTY(GaN, Me,
            MISource("Adachi WILEY 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaN::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G' || point == '*') {
        tMe.c00 = 0.22;
        tMe.c11 = 0.21;
    }
    return tMe;
}

MI_PROPERTY(GaN, Mhh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
Tensor2<double> GaN::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.,0.);
    tMhh.c00 = 1.67;
    tMhh.c11 = 1.64;
    return tMhh;
}

MI_PROPERTY(GaN, Mlh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
Tensor2<double> GaN::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.,0.);
    tMlh.c00 = 1.67;
    tMlh.c11 = 0.15;
    return tMlh;
}

MI_PROPERTY(GaN, CB,
            MISource("-")
           )
double GaN::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point, 'H') + Eg(T,0.,point) );
    return tCB;
    /*if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;stary kod z GaAs*/
}

MI_PROPERTY(GaN, VB,
            MISource("-"),
            MIComment("no temperature dependence")
           )
double GaN::VB(double T, double e, char point, char hole) const {
    double tVB(0.80);
    if (e) {
        /*double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L"); stary kod z GaAs*/
    }
    return tVB;
}

Material::ConductivityType GaN::condtype() const { return Material::CONDUCTIVITY_I; }

bool GaN::isEqual(const Material &other) const {
    return true;
}

Tensor2<double> GaN_bulk::thermk(double T, double t) const {
    return GaN::thermk(T, INFINITY);
}

static MaterialsDB::Register<GaN> materialDB_register_GaN;

static MaterialsDB::Register<GaN_bulk> materialDB_register_GaN_bulk;

}}       // namespace plask::materials
