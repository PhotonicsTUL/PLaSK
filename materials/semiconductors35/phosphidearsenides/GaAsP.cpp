#include "GaAsP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaAsP::GaAsP(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    P = Comp.find("P")->second;
}

std::string GaAsP::str() const { return StringBuilder("Ga")("As")("P", P); }

std::string GaAsP::name() const { return NAME; }

MI_PROPERTY(GaAsP, lattC,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mGaAs.lattC(T,'a') + P*mGaP.lattC(T,'a');
    else if (x == 'c') tLattC = As*mGaAs.lattC(T,'a') + P*mGaP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaAsP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaAsP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mGaAs.Eg(T,e,point) + P*mGaP.Eg(T,e,point) - As*P*0.19;
    else if (point == 'X') tEg = As*mGaAs.Eg(T,e,point) + P*mGaP.Eg(T,e,point) - As*P*0.24;
    else if (point == 'L') tEg = As*mGaAs.Eg(T,e,point) + P*mGaP.Eg(T,e,point) - As*P*0.16;
    else if (point == '*')
    {
        double tEgG = As*mGaAs.Eg(T,e,'G') + P*mGaP.Eg(T,e,'G') - As*P*0.19;
        double tEgX = As*mGaAs.Eg(T,e,'X') + P*mGaP.Eg(T,e,'X') - As*P*0.24;
        double tEgL = As*mGaAs.Eg(T,e,'L') + P*mGaP.Eg(T,e,'L') - As*P*0.16;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaAsP, Dso,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::Dso(double T, double e) const {
    return ( As*mGaAs.Dso(T, e) + P*mGaP.Dso(T, e) );
}

MI_PROPERTY(GaAsP, Me,
            MISource("linear interpolation: GaAs, GaP")
            )
Tensor2<double> GaAsP::Me(double T, double e, char point) const {
    double lMe = As*mGaAs.Me(T,e,point).c00 + P*mGaP.Me(T,e,point).c00 ,
           vMe = As*mGaAs.Me(T,e,point).c11 + P*mGaP.Me(T,e,point).c11 ;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaAsP, Mhh,
            MISource("linear interpolation: GaAs, GaP")
            )
Tensor2<double> GaAsP::Mhh(double T, double e) const {
    double lMhh = As*mGaAs.Mhh(T,e).c00 + P*mGaP.Mhh(T,e).c00,
           vMhh = As*mGaAs.Mhh(T,e).c11 + P*mGaP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaAsP, Mlh,
            MISource("linear interpolation: GaAs, GaP")
            )
Tensor2<double> GaAsP::Mlh(double T, double e) const {
    double lMlh = As*mGaAs.Mlh(T,e).c00 + P*mGaP.Mlh(T,e).c00,
           vMlh = As*mGaAs.Mlh(T,e).c11 + P*mGaP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaAsP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaAsP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaAsP, VB,
            MISource("nonlinear interpolation: GaAs, GaP")
            )
double GaAsP::VB(double T, double e, char point, char hole) const {
    double tVB( As*mGaAs.VB(T,0.,point,hole) + P*mGaP.VB(T,0.,point,hole) - As*P*(-1.06) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
}

MI_PROPERTY(GaAsP, ac,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::ac(double T) const {
    return ( As*mGaAs.ac(T) + P*mGaP.ac(T) );
}

MI_PROPERTY(GaAsP, av,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::av(double T) const {
    return ( As*mGaAs.av(T) + P*mGaP.av(T) );
}

MI_PROPERTY(GaAsP, b,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::b(double T) const {
    return ( As*mGaAs.b(T) + P*mGaP.b(T) );
}

MI_PROPERTY(GaAsP, d,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::d(double T) const {
    return ( As*mGaAs.d(T) + P*mGaP.d(T) );
}

MI_PROPERTY(GaAsP, c11,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::c11(double T) const {
    return ( As*mGaAs.c11(T) + P*mGaP.c11(T) );
}

MI_PROPERTY(GaAsP, c12,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::c12(double T) const {
    return ( As*mGaAs.c12(T) + P*mGaP.c12(T) );
}

MI_PROPERTY(GaAsP, c44,
            MISource("linear interpolation: GaAs, GaP")
            )
double GaAsP::c44(double T) const {
    return ( As*mGaAs.c44(T) + P*mGaP.c44(T) );
}

MI_PROPERTY(GaAsP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> GaAsP::thermk(double T, double t) const {
    double lCondT = 1./(As/mGaAs.thermk(T,t).c00 + P/mGaP.thermk(T,t).c00 + As*P*0.25),
           vCondT = 1./(As/mGaAs.thermk(T,t).c11 + P/mGaP.thermk(T,t).c11 + As*P*0.25);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaAsP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: GaAs, GaP"),
            MIComment("no temperature dependence")
            )
double GaAsP::dens(double T) const {
    return ( As*mGaAs.dens(T) + P*mGaP.dens(T) );
}

MI_PROPERTY(GaAsP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: GaAs, GaP"),
            MIComment("no temperature dependence")
            )
double GaAsP::cp(double T) const {
    return ( As*mGaAs.cp(T) + P*mGaP.cp(T) );
}

MI_PROPERTY(GaAsP, nr,
            MIComment("TODO")
            )
double GaAsP::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(GaAsP, absp,
            MIComment("TODO")
            )
double GaAsP::absp(double wl, double T) const {
    return ( 0. );
}

bool GaAsP::isEqual(const Material &other) const {
    const GaAsP& o = static_cast<const GaAsP&>(other);
    return o.P == this->P;
}

static MaterialsDB::Register<GaAsP> materialDB_register_GaAsP;

}} // namespace plask::materials
