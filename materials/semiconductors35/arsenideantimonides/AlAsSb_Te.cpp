#include "AlAsSb_Te.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlAsSb_Te::name() const { return NAME; }

std::string AlAsSb_Te::str() const { return StringBuilder("Al")("As")("Sb", Sb).dopant("Si", ND); }

MI_PARENT(AlAsSb_Te, AlAsSb)

AlAsSb_Te::AlAsSb_Te(const Material::Composition& Comp, DopingAmountType Type, double Val): AlAsSb(Comp)//, mGaAs_Si(Type,Val), mAlAs_Si(Type,Val)
{
    if (Type == CARRIER_CONCENTRATION)
        Nf_RT = Val;
    else
    {
        ND = Val;
        // taken from n-GaSb:Te - use it for low doping conc
        if (ND <= 1e18)
            Nf_RT = ND;
        else // taken from n-GaSb:Te
        {
            double tNL = log10(ND);
            double tnL = 0.499626*tNL*tNL*tNL - 28.7231*tNL*tNL + 549.517*tNL - 3480.87;
            Nf_RT = ( pow(10.,tnL) );
        }
    }
    double mob_RT_AlAs = 30e-4 + (310e-4 - 30e-4) / (1.+pow(ND/8e17,2.)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    double mob_RT_AlSb = 30e-4 + (200e-4 - 30e-4) / (1.+pow(ND/4e17,3.25)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    mob_RT = 1. / (As/mob_RT_AlAs + Sb/mob_RT_AlSb - 9.3e-3*As*Sb); // for small amount of arsenide
}

MI_PROPERTY(AlAsSb_Te, mob,
            MISource("Strin 1966")
            )
Tensor2<double> AlAsSb_Te::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.8);
    return ( Tensor2<double>(tmob, tmob) );
}

MI_PROPERTY(AlAsSb_Te, Nf,
            MISource("-"),
            MIComment("TODO")
            )
double AlAsSb_Te::Nf(double T) const {
    return ( Nf_RT );
}

double AlAsSb_Te::Dop() const {
    return ( ND );
}

Tensor2<double> AlAsSb_Te::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * mob(T).c00;
    return ( Tensor2<double>(tCond, tCond) );
}

MI_PROPERTY(AlAsSb_Te, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211"),
            MIArgumentRange(MaterialInfo::wl, 500, 7000),
            MIComment("TODO")
            )
double AlAsSb_Te::nr(double wl, double T, double n) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double tE0 = 3.2;
    double tEd = 28.;
    double tEG = 2.338;
    double nR300K2 = 1. + tEd/tE0 + tEd*tE*tE/pow(tE0,3.) + tEd*pow(tE,4.)/(2.*pow(tE0,3.)*(tE0*tE0-tEG*tEG)) * log((2.*tE0*tE0-tEG*tEG-tE*tE)/(tEG*tEG-tE*tE));

    double nR300K;
    if (nR300K2>0) nR300K = sqrt(nR300K2);
    else nR300K = 1.; // TODO
    //taken from n-GaSb
    double nR = nR300K; // TODO // for E << Eg: dnR/dn = 0
    double dnRdT = As*4.6e-5 + Sb*1.19e-5; // from Adachi (2005) ebook p.243 tab. 10.6
    return ( nR + nR*dnRdT*(T-300.) );
}

MI_PROPERTY(AlAsSb, absp,
            MISource("H. Hattasan (2013)"),
            //MIArgumentRange(MaterialInfo::wl, 2000, 20000),
            MIComment("temperature dependence - assumed: (1/abs)(dabs/dT)=1e-3"),
            MIComment("only free-carrier absorption assumed")
            )
double AlAsSb_Te::absp(double wl, double T) const {
    double tAbs_RT = 1.9e-24 * Nf_RT * pow(wl,2.);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool AlAsSb_Te::isEqual(const Material &other) const {
    const AlAsSb_Te& o = static_cast<const AlAsSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAsSb::isEqual(other);
}

static MaterialsDB::Register<AlAsSb_Te> materialDB_register_AlAsSb_Te;

}}       // namespace plask::materials
