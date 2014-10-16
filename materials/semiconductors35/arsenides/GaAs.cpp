#include "GaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {


MI_PROPERTY(GaAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double GaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.65325 + 3.88e-5 * (T-300.);
    return tLattC;
}


MI_PROPERTY(GaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double GaAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(1.519, 0.5405e-3, 204., T);
    else if (point == 'X') tEg = phys::Varshni(1.981, 0.460e-3, 204., T);
    else if (point == 'L') tEg = phys::Varshni(1.815, 0.605e-3, 204., T);
    else if (point == '*')
    {
        double tEgG = phys::Varshni(1.519, 0.5405e-3, 204., T);
        double tEgX = phys::Varshni(1.981, 0.460e-3, 204., T);
        double tEgL = phys::Varshni(1.815, 0.605e-3, 204., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );

}


MI_PROPERTY(GaAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::Dso(double T, double e) const {
    return 0.341;
}


MI_PROPERTY(GaAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.067;
        tMe.c11 = 0.067;
    }
    return tMe;
}


MI_PROPERTY(GaAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaAs::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.33, 0.33); // [001]
    return tMhh;
}


MI_PROPERTY(GaAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaAs::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.090, 0.090);
    return tMlh;
}


MI_PROPERTY(GaAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double GaAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}


MI_PROPERTY(GaAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::VB(double T, double e, char point, char hole) const {
    double tVB(-0.80);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
    return tVB;
}


MI_PROPERTY(GaAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::ac(double T) const {
    return -7.17;
}


MI_PROPERTY(GaAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::av(double T) const {
    return 1.16;
}


MI_PROPERTY(GaAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::b(double T) const {
    return -2.0;
}


MI_PROPERTY(GaAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::d(double T) const {
    return -4.8;
}


MI_PROPERTY(GaAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::c11(double T) const {
    return 122.1;
}


MI_PROPERTY(GaAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::c12(double T) const {
    return 56.6;
}


MI_PROPERTY(GaAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::c44(double T) const {
    return 60.0;
}


MI_PROPERTY(GaAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 150, 1500)
           )
Tensor2<double> GaAs::thermk(double T, double t) const {
    double tCondT = 45.*pow((300./T),1.28);
    return Tensor2<double>(tCondT, tCondT);
}


MI_PROPERTY(GaAs, cond,
            MISource("http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/electric.html"),
            MIComment("Carrier concentration estimated")
           )
Tensor2<double> GaAs::cond(double T) const {
    double c = 1e2 * phys::qe * 8000.* pow((300./T), 2./3.) * 1e16;
    return Tensor2<double>(c, c);
}


MI_PROPERTY(GaAs, nr,
            MISource("D.T.F. Marple, J. Appl. Phys. 35 (1964) 1241-1242"),
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            MIComment("fit by Lukasz Piskorski")
           )
double GaAs::nr(double wl, double T, double n) const {
    double L2 = wl*wl*1e-6;
    double nR296K = sqrt(1.+9.659*L2/(L2-0.137));
    return ( nR296K + nR296K*4.5e-5*(T-296.) );
}


MI_PROPERTY(GaAs, absp,
            MISource(""),
            MIComment("TODO")
           )
double GaAs::absp(double wl, double T) const {
    return 0.;
}


MI_PROPERTY(GaAs, eps,
            MISource("http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/basic.html")
           )
double GaAs::eps(double T) const {
    return 12.9;
}

std::string GaAs::name() const { return NAME; }

bool GaAs::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<GaAs> materialDB_register_GaAs;

}} // namespace plask::materials
