#include "GaInSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInSb::GaInSb(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInSb::str() const { return StringBuilder("Ga")("In", In)("Sb"); }

std::string GaInSb::name() const { return NAME; }

MI_PROPERTY(GaInSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb")
            )
double GaInSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*mGaSb.lattC(T,'a') + In*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*mGaSb.lattC(T,'c') + In*mInSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(GaInSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, InSb")
            )
double GaInSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Ga*In*(0.415);
    else if (point == 'X') tEg = Ga*mGaSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Ga*In*(0.33);
    else if (point == 'L') tEg = Ga*mGaSb.Eg(T,e,point) + In*mInSb.Eg(T,e,point) - Ga*In*(0.4);
    else if (point == '*')
    {
        double tEgG = Ga*mGaSb.Eg(T,e,'G') + In*mInSb.Eg(T,e,'G') - Ga*In*(0.415);
        double tEgX = Ga*mGaSb.Eg(T,e,'X') + In*mInSb.Eg(T,e,'X') - Ga*In*(0.33);
        double tEgL = Ga*mGaSb.Eg(T,e,'L') + In*mInSb.Eg(T,e,'L') - Ga*In*(0.4);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaInSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::Dso(double T, double e) const {
    return ( Ga*mGaSb.Dso(T,e) + In*mInSb.Dso(T,e) - Ga*In*0.1 );
}

MI_PROPERTY(GaInSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L'))
    {
        tMe.c00 = Ga*mGaSb.Me(T,e,point).c00 + In*mInSb.Me(T,e,point).c00,
        tMe.c11 = Ga*mGaSb.Me(T,e,point).c11 + In*mInSb.Me(T,e,point).c11;

    }
    else if (point == '*')
    {
        char pointM = 'G';
        if      ( Eg(T,e,'X') == Eg(T,e,'*') ) pointM = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) pointM = 'L';
        tMe.c00 = Ga*mGaSb.Me(T,e,pointM).c00 + In*mInSb.Me(T,e,pointM).c00;
        tMe.c11 = Ga*mGaSb.Me(T,e,pointM).c11 + In*mInSb.Me(T,e,pointM).c11;
    }
    return ( tMe );
    /*
    double lMe = Ga*mGaSb.Me(T,e,point).c00 + In*mInSb.Me(T,e,point).c00 - Ga*In*0.010,
           vMe = Ga*mGaSb.Me(T,e,point).c11 + In*mInSb.Me(T,e,point).c11 - Ga*In*0.010;
    return ( Tensor2<double>(lMe,vMe) );*/
}

MI_PROPERTY(GaInSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInSb::Mhh(double T, double e) const {
    double lMhh = Ga*mGaSb.Mhh(T,e).c00 + In*mInSb.Mhh(T,e).c00,
           vMhh = Ga*mGaSb.Mhh(T,e).c11 + In*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("nonlinear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInSb::Mlh(double T, double e) const {
    double lMlh = Ga*mGaSb.Mlh(T,e).c00 + In*mInSb.Mlh(T,e).c00 - Ga*In*0.015,
           vMlh = Ga*mGaSb.Mlh(T,e).c11 + In*mInSb.Mlh(T,e).c11 - Ga*In*0.015;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaInSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaInSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*mGaSb.VB(T,0.,point,hole) + In*mInSb.VB(T,0.,point,hole) );
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
    }
    return tVB;
}

MI_PROPERTY(GaInSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::ac(double T) const {
    return ( Ga*mGaSb.ac(T) + In*mInSb.ac(T) );
}

MI_PROPERTY(GaInSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::av(double T) const {
    return ( Ga*mGaSb.av(T) + In*mInSb.av(T) );
}

MI_PROPERTY(GaInSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::b(double T) const {
    return ( Ga*mGaSb.b(T) + In*mInSb.b(T) );
}

MI_PROPERTY(GaInSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::d(double T) const {
    return ( Ga*mGaSb.d(T) + In*mInSb.d(T) );
}

MI_PROPERTY(GaInSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::c11(double T) const {
    return ( Ga*mGaSb.c11(T) + In*mInSb.c11(T) );
}

MI_PROPERTY(GaInSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::c12(double T) const {
    return ( Ga*mGaSb.c12(T) + In*mInSb.c12(T) );
}

MI_PROPERTY(GaInSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::c44(double T) const {
    return ( Ga*mGaSb.c44(T) + In*mInSb.c44(T) );
}

MI_PROPERTY(GaInSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion od nonlinear interpolation of resistivity: GaSb, InSb")
            )
Tensor2<double> GaInSb::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaSb.thermk(T,t).c00 + In/mInSb.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaSb.thermk(T,t).c11 + In/mGaSb.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::dens(double T) const {
    return ( Ga*mGaSb.dens(T) + In*mInSb.dens(T) );
}

MI_PROPERTY(GaInSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaSb, InSb"),
            MIComment("no temperature dependence")
            )
double GaInSb::cp(double T) const {
    return ( Ga*mGaSb.cp(T) + In*mInSb.cp(T) );
}

MI_PROPERTY(GaInSb, nr,
            MIComment("TODO")
            )
double GaInSb::nr(double wl, double T, double n) const {
    throw NotImplemented("nr for GaInSb");
}

MI_PROPERTY(GaInSb, absp,
            MIComment("TODO")
            )
double GaInSb::absp(double wl, double T) const {
    throw NotImplemented("absp for GaInSb");
}

bool GaInSb::isEqual(const Material &other) const {
    const GaInSb& o = static_cast<const GaInSb&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInSb> materialDB_register_GaInSb;

}} // namespace plask::materials
