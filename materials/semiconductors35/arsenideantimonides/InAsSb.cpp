#include "InAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

InAsSb::InAsSb(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string InAsSb::str() const { return StringBuilder("In")("As")("Sb", Sb); }

std::string InAsSb::name() const { return NAME; }

MI_PROPERTY(InAsSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mInAs.lattC(T,'a') + Sb*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = As*mInAs.lattC(T,'a') + Sb*mInSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(InAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: InAs, InSb")
            )
double InAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mInAs.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - As*Sb*0.67;
    else if (point == 'X') tEg = As*mInAs.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - As*Sb*0.6;
    else if (point == 'L') tEg = As*mInAs.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - As*Sb*0.6;
    else if (point == '*') {
        double tEgG = As*mInAs.Eg(T,e,'G') + Sb*mInSb.Eg(T,e,'G') - As*Sb*0.67;
        double tEgX = As*mInAs.Eg(T,e,'X') + Sb*mInSb.Eg(T,e,'X') - As*Sb*0.6;
        double tEgL = As*mInAs.Eg(T,e,'L') + Sb*mInSb.Eg(T,e,'L') - As*Sb*0.6;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InAsSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::Dso(double T, double e) const {
    return ( As*mInAs.Dso(T, e) + Sb*mInSb.Dso(T, e) - As*Sb*1.2 );
}

MI_PROPERTY(InAsSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAsSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = As*mInAs.Me(T,e,point).c00 + Sb*mInSb.Me(T,e,point).c00,
        tMe.c11 = As*mInAs.Me(T,e,point).c11 + Sb*mInSb.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if      ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = As*mInAs.Me(T,e,point).c00 + Sb*mInSb.Me(T,e,point).c00;
        tMe.c11 = As*mInAs.Me(T,e,point).c11 + Sb*mInSb.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -As*Sb*(0.027) );
        tMe.c11 += ( -As*Sb*(0.027) );
    }
    return ( tMe );
}

MI_PROPERTY(InAsSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAsSb::Mhh(double T, double e) const {
    double lMhh = As*mInAs.Mhh(T,e).c00 + Sb*mInSb.Mhh(T,e).c00,
           vMhh = As*mInAs.Mhh(T,e).c11 + Sb*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(InAsSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> InAsSb::Mlh(double T, double e) const {
    double lMlh = As*mInAs.Mlh(T,e).c00 + Sb*mInSb.Mlh(T,e).c00,
           vMlh = As*mInAs.Mlh(T,e).c11 + Sb*mInSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(InAsSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> InAsSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(InAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(InAsSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( As*mInAs.VB(T,0.,point,hole) + Sb*mInSb.VB(T,0.,point,hole) );
    if (!e) return tVB;
    else {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(InAsSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::ac(double T) const {
    return ( As*mInAs.ac(T) + Sb*mInSb.ac(T) );
}

MI_PROPERTY(InAsSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::av(double T) const {
    return ( As*mInAs.av(T) + Sb*mInSb.av(T) );
}

MI_PROPERTY(InAsSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::b(double T) const {
    return ( As*mInAs.b(T) + Sb*mInSb.b(T) );
}

MI_PROPERTY(InAsSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::d(double T) const {
    return ( As*mInAs.d(T) + Sb*mInSb.d(T) );
}

MI_PROPERTY(InAsSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::c11(double T) const {
    return ( As*mInAs.c11(T) + Sb*mInSb.c11(T) );
}

MI_PROPERTY(InAsSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::c12(double T) const {
    return ( As*mInAs.c12(T) + Sb*mInSb.c12(T) );
}

MI_PROPERTY(InAsSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::c44(double T) const {
    return ( As*mInAs.c44(T) + Sb*mInSb.c44(T) );
}

MI_PROPERTY(InAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: InAs, InSb")
            )
Tensor2<double> InAsSb::thermk(double T, double t) const {
    double lCondT = 1./(As/mInAs.thermk(T,t).c00 + Sb/mInSb.thermk(T,t).c00 + As*Sb*0.91),
           vCondT = 1./(As/mInAs.thermk(T,t).c11 + Sb/mInSb.thermk(T,t).c11 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(InAsSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::dens(double T) const {
    return ( As*mInAs.dens(T) + Sb*mInSb.dens(T) );
}

MI_PROPERTY(InAsSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::cp(double T) const {
    return ( As*mInAs.cp(T) + Sb*mInSb.cp(T) );
}

Material::ConductivityType InAsSb::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(InAsSb, nr,
            MISource("P.P. Paskov et al., J. Appl. Phys. 81 (1997) 1890-1898; "), // nR @ RT
            MISource("linear interpolation: InAs(0.80)Sb(0.20), InSb"),
            MIComment("nr(wv) relations for interpolation fitted by L. Piskorski (PLaSK developer), unpublished; "),
            MIComment("do not use for InAsSb with Sb content higher than 0.20"),
            MIArgumentRange(MaterialInfo::wl, 2050, 3450)
            )
double InAsSb::nr(double wl, double T, double n) const {
    double nR_InAs080Sb020_300K = 0.01525*pow(wl*1e-3,1.783)+3.561; // 2.05 um < wl < 5.4 um
    double nR_InAs_300K = 2.873e-5*pow(wl*1e-3,6.902)+3.438; // 2.05 um < wl < 3.45 um
    double v = 5.*As-4;
    double nR300K = v*nR_InAs_300K + (1.-v)*nR_InAs080Sb020_300K;

    double nR = nR300K;

    double dnRdT = As*12e-5 + Sb*6.9e-5; // from Adachi (2005) ebook p.243 tab. 10.6

    return ( nR + nR*dnRdT*(T-300.) );
}

MI_PROPERTY(InAsSb, absp,
            MIComment("TODO")
            )
double InAsSb::absp(double wl, double T) const {
    throw NotImplemented("absp for InAsSb");
}

bool InAsSb::isEqual(const Material &other) const {
    const InAsSb& o = static_cast<const InAsSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<InAsSb> materialDB_register_InAsSb;

}} // namespace plask::materials
