#include "GaInP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInP::GaInP(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
}

std::string GaInP::str() const { return StringBuilder("In", In)("Ga")("P"); }

std::string GaInP::name() const { return NAME; }

MI_PROPERTY(GaInP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*mGaP.lattC(T,'a') + In*mInP.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*mGaP.lattC(T,'a') + In*mInP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaInP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: GaP, InP")
            )
double GaInP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*mGaP.Eg(T, e, point) + In*mInP.Eg(T, e, point) - Ga*In*0.65;
    else if (point == 'X') tEg = Ga*mGaP.Eg(T, e, point) + In*mInP.Eg(T, e, point) - Ga*In*0.20;
    else if (point == 'L') tEg = Ga*mGaP.Eg(T, e, point) + In*mInP.Eg(T, e, point) - Ga*In*1.03;
    else if (point == '*')
    {
        double tEgG = Ga*mGaP.Eg(T, e, 'G') + In*mInP.Eg(T, e, 'G') - Ga*In*0.65;
        double tEgX = Ga*mGaP.Eg(T, e, 'X') + In*mInP.Eg(T, e, 'X') - Ga*In*0.20;
        double tEgL = Ga*mGaP.Eg(T, e, 'L') + In*mInP.Eg(T, e, 'L') - Ga*In*1.03;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaInP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::Dso(double T, double e) const {
    return ( Ga*mGaP.Dso(T,e) + In*mInP.Dso(T,e) );
}

MI_PROPERTY(GaInP, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInP::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Ga*mGaP.Me(T,e,point).c00 + In*mInP.Me(T,e,point).c00,
        tMe.c11 = Ga*mGaP.Me(T,e,point).c11 + In*mInP.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Ga*mGaP.Me(T,e,point).c00 + In*mInP.Me(T,e,point).c00;
        tMe.c11 = Ga*mGaP.Me(T,e,point).c11 + In*mInP.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -Ga*In*(0.01854) );
        tMe.c11 += ( -Ga*In*(0.01854) );
    }
    return ( tMe );
}

MI_PROPERTY(GaInP, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInP::Mhh(double T, double e) const {
    double lMhh = Ga*mGaP.Mhh(T,e).c00 + In*mInP.Mhh(T,e).c00,
           vMhh = Ga*mGaP.Mhh(T,e).c11 + In*mInP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInP, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaInP::Mlh(double T, double e) const {
    double lMlh = Ga*mGaP.Mlh(T,e).c00 + In*mInP.Mlh(T,e).c00,
           vMlh = Ga*mGaP.Mlh(T,e).c11 + In*mInP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInP, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaInP::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaInP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*mGaP.VB(T,0.,point,hole) + In*mInP.VB(T,0.,point,hole) );
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

MI_PROPERTY(GaInP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::ac(double T) const {
    return ( Ga*mGaP.ac(T) + In*mInP.ac(T) );
}

MI_PROPERTY(GaInP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::av(double T) const {
    return ( Ga*mGaP.av(T) + In*mInP.av(T) );
}

MI_PROPERTY(GaInP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::b(double T) const {
    return ( Ga*mGaP.b(T) + In*mInP.b(T) );
}

MI_PROPERTY(GaInP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::d(double T) const {
    return ( Ga*mGaP.d(T) + In*mInP.d(T) );
}

MI_PROPERTY(GaInP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP")
            )
double GaInP::c11(double T) const {
    return ( Ga*mGaP.c11(T) + In*mInP.c11(T) );
}

MI_PROPERTY(GaInP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::c12(double T) const {
    return ( Ga*mGaP.c12(T) + In*mInP.c12(T) );
}

MI_PROPERTY(GaInP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::c44(double T) const {
    return ( Ga*mGaP.c44(T) + In*mInP.c44(T) );
}

MI_PROPERTY(GaInP, A,
            MISource("L. Piskorski, Analiza numeryczna polprzewodnikowego lasera zlaczowego typu VCSEL emitujacego promieniowanie o dlugości fali dostosowanej do systemow telekomunikacyjnych bazujacych na swiatlowodach plastikowych (in Polish), MSc thesis, 2006"),
            MIComment("no temperature dependence")
            )
double GaInP::A(double T) const {
    return ( 1e8 );
}

MI_PROPERTY(GaInP, B,
            MISource("L. Piskorski, Analiza numeryczna polprzewodnikowego lasera zlaczowego typu VCSEL emitujacego promieniowanie o dlugości fali dostosowanej do systemow telekomunikacyjnych bazujacych na swiatlowodach plastikowych (in Polish), MSc thesis, 2006")
            )
double GaInP::B(double T) const {
    return ( 1e-10*pow(300/T,1.5) );
}

MI_PROPERTY(GaInP, C,
            MISource("W. W. Chow et al., IEEE Journal of Selected Topics in Quantum Electronics 1 (1995) 649-653"),
            MIComment("no temperature dependence")
            )
double GaInP::C(double T) const {
    return ( 3.5e-30 );
}

MI_PROPERTY(GaInP, D,
            MISource("O. Imafuji et al., Journal of Selected Topics in Quantum Electronics 5 (1999) 721-728"), // D(300K)
            MISource("L. Piskorski, Analiza numeryczna polprzewodnikowego lasera zlaczowego typu VCSEL emitujacego promieniowanie o dlugości fali dostosowanej do systemow telekomunikacyjnych bazujacych na swiatlowodach plastikowych (in Polish), MSc thesis, 2006") // D(T)
            )
double GaInP::D(double T) const {
    return ( T/300. );
}

MI_PROPERTY(GaInP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: GaP, InP")
            )
Tensor2<double> GaInP::thermk(double T, double t) const {
    double lCondT = 1./(Ga/mGaP.thermk(T,t).c00 + In/mInP.thermk(T,t).c00 + Ga*In*0.72),
           vCondT = 1./(Ga/mGaP.thermk(T,t).c11 + In/mInP.thermk(T,t).c11 + Ga*In*0.72);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::dens(double T) const {
    return ( Ga*mGaP.dens(T) + In*mInP.dens(T) );
}

MI_PROPERTY(GaInP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaP, InP"),
            MIComment("no temperature dependence")
            )
double GaInP::cp(double T) const {
    return ( Ga*mGaP.cp(T) + In*mInP.cp(T) );
}

MI_PROPERTY(GaInP, nr,
            MIComment("TODO")
            )
double GaInP::nr(double wl, double T, double n) const {
    throw NotImplemented("nr for GaInP");
}

MI_PROPERTY(GaInP, absp,
            MIComment("TODO")
            )
double GaInP::absp(double wl, double T) const {
    throw NotImplemented("absp for GaInP");
}

bool GaInP::isEqual(const Material &other) const {
    const GaInP& o = static_cast<const GaInP&>(other);
    return o.Ga == this->Ga;
}

static MaterialsDB::Register<GaInP> materialDB_register_GaInP;

}} // namespace plask::materials
