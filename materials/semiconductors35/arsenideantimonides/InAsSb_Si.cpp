#include "InAsSb_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAsSb_Si::name() const { return NAME; }

std::string InAsSb_Si::str() const { return StringBuilder("In")("As")("Sb", Sb).dopant("Si", ND); }

MI_PARENT(InAsSb_Si, InAsSb)

InAsSb_Si::InAsSb_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): InAsSb(Comp)//, mGaAs_Si(Type,Val), mAlAs_Si(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        ND = Val;
        // Barashar (2010) only data for InAs used - use it for InAsSb with high As content
        if (ND <= 1e19)
            Nf_RT = ND;
        else
        {
            double tNL = log10(ND);
            double tnL = -0.259963*tNL*tNL + 10.9705*tNL - 95.5924;
            Nf_RT = ( pow(10.,tnL) );
        }
    }
    double tInAs_mob_RT = 450. + (12000. - 450.) / (1.+pow(ND/2e18,0.80));
    mob_RT = tInAs_mob_RT; // data for InAs(0.91)Sb(0.09) fit to above relation (see: Talercio 2014)
}

MI_PROPERTY(InAsSb_Si, mob,
            MISource("T. Taliercio, Optics Express 22 (2014) pp. 24294-24303"),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("fitted for high doping and high As content"),
            MIComment("mob(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAsSb_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.7); // Adachi (2005) book (d=1.7 for InAs)
    return ( Tensor2<double>(tmob, tmob) );
}

MI_PROPERTY(InAsSb_Si, Nf,
            MISource("T. Taliercio, Optics Express 22 (2014) pp. 24294-24303"),
            MIComment("fit by Lukasz Piskorski"),
            MIComment("no temperature dependence")
            )
double InAsSb_Si::Nf(double T) const {
    double tD = -0.00332*log10(Nf_RT)+0.26;
    return ( Nf_RT*pow(T/300.,tD) );
}

double InAsSb_Si::Dop() const {
    return ( ND );
}

MI_PROPERTY(InAsSb_Si, cond,
            MIComment("cond(N,T) = q * n(N,T) * mob(n(N,T),T)")
            )
Tensor2<double> InAsSb_Si::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * (mob(T).c00)*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(InAsSb_Si, nr,
            MISource("P.P. Paskov et al., J. Appl. Phys. 81 (1997) 1890-1898; "), // nR @ RT
            MISource("linear interpolation: InAs(0.80)Sb(0.20), InSb"),
            MIComment("nr(wv) relations for interpolation fitted by L. Piskorski (PLaSK developer), unpublished; "),
            MIComment("do not use for InAsSb with Sb content higher than 0.20"),
            MIArgumentRange(MaterialInfo::wl, 2050, 3450)
            )
double InAsSb_Si::nr(double wl, double T, double n) const {
    double nR_InAs080Sb020_300K = 0.01525*pow(wl*1e-3,1.783)+3.561; // 2.05 um < wl < 5.4 um
    double nR_InAs_300K = 2.873e-5*pow(wl*1e-3,6.902)+3.438; // 2.05 um < wl < 3.45 um
    double v = 5.*As-4;
    double nR300K = v*nR_InAs_300K + (1.-v)*nR_InAs080Sb020_300K;

    double dnRn = 0.; // influence of free carrier conc.
    if (Nf_RT >= 3.59e16) dnRn = -0.06688*pow((log10(Nf_RT)),2.) + 2.18936*log10(Nf_RT) -17.9151;
    double nR = nR300K + dnRn;

    double dnRdT = As*12e-5 + Sb*6.9e-5; // from Adachi (2005) ebook p.243 tab. 10.6

    return ( nR + nR*dnRdT*(T-300.) );
}

MI_PROPERTY(InAsSb, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::wl, 2000, 20000),
            MIComment("use it for highly doped InAsSb with high As content") // energy band narrowing included
            )
double InAsSb_Si::absp(double wl, double T) const {
    double tAbs_RT = 1e24*exp(-wl/33.) + ( As*6.5e-29*Nf_RT*pow(wl,3.) + Sb*2.8e-25*Nf_RT*pow(wl,2.) ) + pow(20.*sqrt(Nf_RT*1e-18),1.05);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool InAsSb_Si::isEqual(const Material &other) const {
    const InAsSb_Si& o = static_cast<const InAsSb_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAsSb::isEqual(other);
}

static MaterialsDB::Register<InAsSb_Si> materialDB_register_InAsSb_Si;

}}       // namespace plask::materials
