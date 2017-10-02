#include "InAsP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

InAsP::InAsP(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    P = Comp.find("P")->second;
}

std::string InAsP::str() const { return StringBuilder("In")("As")("P", P); }

std::string InAsP::name() const { return NAME; }

MI_PROPERTY(InAsP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP")
            )
double InAsP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mInAs.lattC(T,'a') + P*mInP.lattC(T,'a');
    else if (x == 'c') tLattC = As*mInAs.lattC(T,'a') + P*mInP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(InAsP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: InAs, InP")
            )
double InAsP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mInAs.Eg(T,e,point) + P*mInP.Eg(T,e,point) - As*P*0.10;
    else if (point == 'X') tEg = As*mInAs.Eg(T,e,point) + P*mInP.Eg(T,e,point) - As*P*0.27;
    else if (point == 'L') tEg = As*mInAs.Eg(T,e,point) + P*mInP.Eg(T,e,point) - As*P*0.27;
    else if (point == '*')
    {
        double tEgG = As*mInAs.Eg(T,e,'G') + P*mInP.Eg(T,e,'G') - As*P*0.10;
        double tEgX = As*mInAs.Eg(T,e,'X') + P*mInP.Eg(T,e,'X') - As*P*0.27;
        double tEgL = As*mInAs.Eg(T,e,'L') + P*mInP.Eg(T,e,'L') - As*P*0.27;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InAsP, Dso,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009; "),
            MISource("nonlinear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::Dso(double T, double e) const {
    return ( As*mInAs.Dso(T, e) + P*mInP.Dso(T, e) - As*P*0.16 );
}

MI_PROPERTY(InAsP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAsP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = As*mInAs.Me(T,e,point).c00 + P*mInP.Me(T,e,point).c00,
        tMe.c11 = As*mInAs.Me(T,e,point).c11 + P*mInP.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = As*mInAs.Me(T,e,point).c00 + P*mInP.Me(T,e,point).c00;
        tMe.c11 = As*mInAs.Me(T,e,point).c11 + P*mInP.Me(T,e,point).c11;
    }
    return ( tMe );
}

MI_PROPERTY(InAsP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAsP::Mhh(double T, double e) const {
    double lMhh = As*mInAs.Mhh(T,e).c00 + P*mInP.Mhh(T,e).c00,
           vMhh = As*mInAs.Mhh(T,e).c11 + P*mInP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(InAsP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAsP::Mlh(double T, double e) const {
    double lMlh = As*mInAs.Mlh(T,e).c00 + P*mInP.Mlh(T,e).c00,
           vMlh = As*mInAs.Mlh(T,e).c11 + P*mInP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(InAsP, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> InAsP::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(InAsP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InAsP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(InAsP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::VB(double T, double e, char point, char hole) const {
    double tVB( As*mInAs.VB(T,0.,point,hole) + P*mInP.VB(T,0.,point,hole) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(InAsP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::ac(double T) const {
    return ( As*mInAs.ac(T) + P*mInP.ac(T) );
}

MI_PROPERTY(InAsP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::av(double T) const {
    return ( As*mInAs.av(T) + P*mInP.av(T) );
}

MI_PROPERTY(InAsP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::b(double T) const {
    return ( As*mInAs.b(T) + P*mInP.b(T) );
}

MI_PROPERTY(InAsP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::d(double T) const {
    return ( As*mInAs.d(T) + P*mInP.d(T) );
}

MI_PROPERTY(InAsP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::c11(double T) const {
    return ( As*mInAs.c11(T) + P*mInP.c11(T) );
}

MI_PROPERTY(InAsP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::c12(double T) const {
    return ( As*mInAs.c12(T) + P*mInP.c12(T) );
}

MI_PROPERTY(InAsP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::c44(double T) const {
    return ( As*mInAs.c44(T) + P*mInP.c44(T) );
}

MI_PROPERTY(InAsP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: InAs, InP")
            )
Tensor2<double> InAsP::thermk(double T, double t) const {
    double lCondT = 1./(As/mInAs.thermk(T,t).c00 + P/mInP.thermk(T,t).c00 + As*P*0.25),
           vCondT = 1./(As/mInAs.thermk(T,t).c11 + P/mInP.thermk(T,t).c11 + As*P*0.25);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(InAsP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::dens(double T) const {
    return ( As*mInAs.dens(T) + P*mInP.dens(T) );
}

MI_PROPERTY(InAsP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: InAs, InP"),
            MIComment("no temperature dependence")
            )
double InAsP::cp(double T) const {
    return ( As*mInAs.cp(T) + P*mInP.cp(T) );
}

Material::ConductivityType InAsP::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(InAsP, nr,
            MIComment("TODO")
            )
double InAsP::nr(double lam, double T, double n) const {
    throw NotImplemented("nr for InAsP");
}

MI_PROPERTY(InAsP, absp,
            MIComment("TODO")
            )
double InAsP::absp(double lam, double T) const {
    throw NotImplemented("abps for InAsP");
}

bool InAsP::isEqual(const Material &other) const {
    const InAsP& o = static_cast<const InAsP&>(other);
    return o.P == this->P;
}

static MaterialsDB::Register<InAsP> materialDB_register_InAsP;

}} // namespace plask::materials
