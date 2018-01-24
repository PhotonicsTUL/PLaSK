#include "GaInAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInAs::GaInAs(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInAs::str() const { return StringBuilder("In", In)("Ga")("As"); }

std::string GaInAs::name() const { return NAME; }

MI_PROPERTY(GaInAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*mGaAs.lattC(T,'a') + In*mInAs.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*mGaAs.lattC(T,'a') + In*mInAs.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaInAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaAs, InAs")
            )
double GaInAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Ga*In*0.477;
    else if (point == 'X') tEg = Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Ga*In*1.4;
    else if (point == 'L') tEg = Ga*mGaAs.Eg(T,e,point) + In*mInAs.Eg(T,e,point) - Ga*In*0.33;
    else if (point == '*') {
        double tEgG = Ga*mGaAs.Eg(T,e,'G') + In*mInAs.Eg(T,e,'G') - Ga*In*0.477;
        double tEgX = Ga*mGaAs.Eg(T,e,'X') + In*mInAs.Eg(T,e,'X') - Ga*In*1.4;
        double tEgL = Ga*mGaAs.Eg(T,e,'L') + In*mInAs.Eg(T,e,'L') - Ga*In*0.33;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaInAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::Dso(double T, double e) const {
    return ( Ga*mGaAs.Dso(T, e) + In*mInAs.Dso(T, e) - Ga*In*0.15 );
}

MI_PROPERTY(GaInAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Ga*mGaAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00,
        tMe.c11 = Ga*mGaAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Ga*mGaAs.Me(T,e,point).c00 + In*mInAs.Me(T,e,point).c00;
        tMe.c11 = Ga*mGaAs.Me(T,e,point).c11 + In*mInAs.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -Ga*In*(0.008) );
        tMe.c11 += ( -Ga*In*(0.008) );
    }
    return ( tMe );
}

MI_PROPERTY(GaInAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs::Mhh(double T, double e) const {
    double lMhh = Ga*mGaAs.Mhh(T,e).c00 + In*mInAs.Mhh(T,e).c00,
           vMhh = Ga*mGaAs.Mhh(T,e).c11 + In*mInAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInAs::Mlh(double T, double e) const {
    double lMlh = Ga*mGaAs.Mlh(T,e).c00 + In*mInAs.Mlh(T,e).c00,
           vMlh = Ga*mGaAs.Mlh(T,e).c11 + In*mInAs.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaInAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaInAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*mGaAs.VB(T,0.,point,hole) + In*mInAs.VB(T,0.,point,hole) - Ga*In*(-0.38) );
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

MI_PROPERTY(GaInAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::ac(double T) const {
    return ( Ga*mGaAs.ac(T) + In*mInAs.ac(T) - Ga*In*2.61 );
}

MI_PROPERTY(GaInAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::av(double T) const {
    return ( Ga*mGaAs.av(T) + In*mInAs.av(T) );
}

MI_PROPERTY(GaInAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::b(double T) const {
    return ( Ga*mGaAs.b(T) + In*mInAs.b(T) );
}

MI_PROPERTY(GaInAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::d(double T) const {
    return ( Ga*mGaAs.d(T) + In*mInAs.d(T) );
}

MI_PROPERTY(GaInAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs")
            )
double GaInAs::c11(double T) const {
    return ( Ga*mGaAs.c11(T) + In*mInAs.c11(T) );
}

MI_PROPERTY(GaInAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::c12(double T) const {
    return ( Ga*mGaAs.c12(T) + In*mInAs.c12(T) );
}

MI_PROPERTY(GaInAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::c44(double T) const {
    return ( Ga*mGaAs.c44(T) + In*mInAs.c44(T) );
}

MI_PROPERTY(GaInAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: GaAs, InAs")
            )
Tensor2<double> GaInAs::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaAs.thermk(T,t).c00 + In/mInAs.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaAs.thermk(T,t).c11 + In/mInAs.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::dens(double T) const {
    return ( Ga*mGaAs.dens(T) + In*mInAs.dens(T) );
}

MI_PROPERTY(GaInAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaAs, InAs"),
            MIComment("no temperature dependence")
            )
double GaInAs::cp(double T) const {
    return ( Ga*mGaAs.cp(T) + In*mInAs.cp(T) );
}

Material::ConductivityType GaInAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaInAs, nr,
            MISource("S. Adachi, J. Appl. Phys. 53 (1982) 5863-5869; "),
            MISource("J. Piprek et al., IEEE Photon. Technol. Lett. 6 (1994) 139-142; "),
            MISource("D. Dey et al., Appl. Phys. Lett. 94 (2009) 081109"),
            MIArgumentRange(MaterialInfo::lam, 950, 12400)
            )
double GaInAs::nr(double lam, double T, double /*n*/) const {
    double tEf = phys::PhotonEnergy(lam),
           tEg = Eg(T,0,'G'),
           tDs0 = Dso(T,0),
           tA = 9.29 - 4.93*In,
           tB = 7.86 + 2.66*In,
           tChi = tEf / tEg,
           tChi0 = tEf / (tEg + tDs0),
           tFChi = (1/tChi/tChi) * (2 - sqrt(1 + tChi) - sqrt(1 - tChi)),
           tFChi0 = (1/tChi0/tChi0) * (2 - sqrt(1 + tChi0) - sqrt(1 - tChi0)),
           tnr = sqrt(tA * (tFChi + 0.5 * pow(tEg/(tEg + tDs0),1.5) * tFChi0) + tB),
           tBeta(0.);

    if (In > 0.45 && In < 0.55) tBeta = 2e-4; //J. Piprek
    else if (In > 0.6 && In < 0.7) tBeta = 7.8e-4; //D. Dey

    return ( tnr + tBeta*(T - 300.) );
}

MI_PROPERTY(GaInAs, absp,
            MIComment("TODO")
            )
double GaInAs::absp(double /*lam*/, double /*T*/) const {
    throw NotImplemented("absp for GaInAs");
}

double GaInAs::eps(double T) const {
    return In*mInAs.eps(T) + Ga*mGaAs.eps(T);
}

bool GaInAs::isEqual(const Material &other) const {
    const GaInAs& o = static_cast<const GaInAs&>(other);
    return o.Ga == this->Ga;
}

static MaterialsDB::Register<GaInAs> materialDB_register_GaInAs;

}} // namespace plask::materials
