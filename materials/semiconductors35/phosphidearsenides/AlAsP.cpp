#include "AlAsP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlAsP::AlAsP(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    P = Comp.find("P")->second;
}

std::string AlAsP::str() const { return StringBuilder("Al")("As")("P", P); }

std::string AlAsP::name() const { return NAME; }

MI_PROPERTY(AlAsP, lattC,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mAlAs.lattC(T,'a') + P*mAlP.lattC(T,'a');
    else if (x == 'c') tLattC = As*mAlAs.lattC(T,'a') + P*mAlP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(AlAsP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlAsP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mAlAs.Eg(T,e,point) + P*mAlP.Eg(T,e,point) - As*P*0.22;
    else if (point == 'X') tEg = As*mAlAs.Eg(T,e,point) + P*mAlP.Eg(T,e,point) - As*P*0.22;
    else if (point == 'L') tEg = As*mAlAs.Eg(T,e,point) + P*mAlP.Eg(T,e,point) - As*P*0.22;
    else if (point == '*')
    {
        double tEgG = As*mAlAs.Eg(T,e,'G') + P*mAlP.Eg(T,e,'G') - As*P*0.22;
        double tEgX = As*mAlAs.Eg(T,e,'X') + P*mAlP.Eg(T,e,'X') - As*P*0.22;
        double tEgL = As*mAlAs.Eg(T,e,'L') + P*mAlP.Eg(T,e,'L') - As*P*0.22;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlAsP, Dso,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::Dso(double T, double e) const {
    return ( As*mAlAs.Dso(T, e) + P*mAlP.Dso(T, e) );
}

MI_PROPERTY(AlAsP, Me,
            MISource("linear interpolation: AlAs, AlP")
            )
Tensor2<double> AlAsP::Me(double T, double e, char point) const {
    double lMe = As*mAlAs.Me(T,e,point).c00 + P*mAlP.Me(T,e,point).c00 ,
           vMe = As*mAlAs.Me(T,e,point).c11 + P*mAlP.Me(T,e,point).c11 ;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlAsP, Mhh,
            MISource("linear interpolation: AlAs, AlP")
            )
Tensor2<double> AlAsP::Mhh(double T, double e) const {
    double lMhh = As*mAlAs.Mhh(T,e).c00 + P*mAlP.Mhh(T,e).c00,
           vMhh = As*mAlAs.Mhh(T,e).c11 + P*mAlP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlAsP, Mlh,
            MISource("linear interpolation: AlAs, AlP")
            )
Tensor2<double> AlAsP::Mlh(double T, double e) const {
    double lMlh = As*mAlAs.Mlh(T,e).c00 + P*mAlP.Mlh(T,e).c00,
           vMlh = As*mAlAs.Mlh(T,e).c11 + P*mAlP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlAsP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlAsP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlAsP, VB,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::VB(double T, double e, char point, char hole) const {
    double tVB( As*mAlAs.VB(T,0.,point,hole) + P*mAlP.VB(T,0.,point,hole) );
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

MI_PROPERTY(AlAsP, ac,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::ac(double T) const {
    return ( As*mAlAs.ac(T) + P*mAlP.ac(T) );
}

MI_PROPERTY(AlAsP, av,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::av(double T) const {
    return ( As*mAlAs.av(T) + P*mAlP.av(T) );
}

MI_PROPERTY(AlAsP, b,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::b(double T) const {
    return ( As*mAlAs.b(T) + P*mAlP.b(T) );
}

MI_PROPERTY(AlAsP, d,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::d(double T) const {
    return ( As*mAlAs.d(T) + P*mAlP.d(T) );
}

MI_PROPERTY(AlAsP, c11,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::c11(double T) const {
    return ( As*mAlAs.c11(T) + P*mAlP.c11(T) );
}

MI_PROPERTY(AlAsP, c12,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::c12(double T) const {
    return ( As*mAlAs.c12(T) + P*mAlP.c12(T) );
}

MI_PROPERTY(AlAsP, c44,
            MISource("linear interpolation: AlAs, AlP")
            )
double AlAsP::c44(double T) const {
    return ( As*mAlAs.c44(T) + P*mAlP.c44(T) );
}

MI_PROPERTY(AlAsP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> AlAsP::thermk(double T, double t) const {
    double lCondT = 1./(As/mAlAs.thermk(T,t).c00 + P/mAlP.thermk(T,t).c00 + As*P*0.25),
           vCondT = 1./(As/mAlAs.thermk(T,t).c11 + P/mAlP.thermk(T,t).c11 + As*P*0.25);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlAsP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlAs, AlP"),
            MIComment("no temperature dependence")
            )
double AlAsP::dens(double T) const {
    return ( As*mAlAs.dens(T) + P*mAlP.dens(T) );
}

MI_PROPERTY(AlAsP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlAs, AlP"),
            MIComment("no temperature dependence")
            )
double AlAsP::cp(double T) const {
    return ( As*mAlAs.cp(T) + P*mAlP.cp(T) );
}

MI_PROPERTY(AlAsP, nr,
            MIComment("TODO")
            )
double AlAsP::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(AlAsP, absp,
            MIComment("TODO")
            )
double AlAsP::absp(double wl, double T) const {
    return ( 0. );
}

bool AlAsP::isEqual(const Material &other) const {
    const AlAsP& o = static_cast<const AlAsP&>(other);
    return o.P == this->P;
}

static MaterialsDB::Register<AlAsP> materialDB_register_AlAsP;

}} // namespace plask::materials
