#ifndef PLASK__MATERIAL_INFO_H
#define PLASK__MATERIAL_INFO_H

/** @file
This file includes classes which stores meta-informations about materials.
*/

//TODO impl.

#include <string>
#include <map>
#include <vector>
#include <utility>

namespace plask {

class MaterialInfo {

    enum PROPERTY_NAME {
        kind,
        lattC,
        Eg,
        CBO,
        VBO,
        Dso,
        Mso,
        Me,
        Me_v,
        Me_l,
        Mhh,
        Mhh_v,
        Mhh_l,
        Mlh,
        Mlh_v,
        Mlh_l,
        Mh,
        Mh_v,
        Mh_l,
        eps,
        chi,
        Nc,
        Ni,
        Nf,
        EactD,
        EactA,      ///<acceptor ionisation energy
        mob,        ///<mobility
        cond,       ///<electrical conductivity
        cond_v,     ///<electrical conductivity in vertical direction
        cond_l,     ///<electrical conductivity in lateral direction
        res,        ///<electrical resistivity
        res_v,      ///<electrical resistivity in vertical direction
        res_l,      ///<electrical resistivity in lateral direction
        A,          ///<monomolecular recombination coefficient
        B,          ///<radiative recombination coefficient
        C,          ///<Auger recombination coefficient
        D,          ///<ambipolar diffusion coefficient
        condT,      ///<thermal conductivity
        condT_v,    ///<thermal conductivity in vertical direction
        condT_l,    ///<thermal conductivity in lateral direction
        dens,       ///<density
        specHeat,   ///<specific heat at constant pressure
        nr,         ///<refractive index
        absp,       ///<absorption coefficient alpha
        Nr          ///<refractive index
    };

    ///Names of arguments for which range we need give
    enum ARGUMENT_NAME {
        T,          ///<temperature [K]
        thickness,  ///<thickness [m]
        wl          ///<Wavelength [nm]
    };

    ///Name of parent class
    std::string parent;

    struct PropertyInfo {

        typedef std::pair<double, double> ArgumentRange;

        std::map<ARGUMENT_NAME, ArgumentRange> argumentRange;

        std::string source;

        std::string comment;
    };

    std::map<PROPERTY_NAME, PropertyInfo> propertyInfo;

};

class MaterialInfoDB {



public:

    static MaterialInfoDB& getDefault();

    ///Material name -> material information
    std::map<std::string, MaterialInfo> materialInfo;

};

}

#endif // PLASK__MATERIAL_INFO_H
