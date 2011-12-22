#ifndef PLASK__MATERIAL_H
#define PLASK__MATERIAL_H

/** @file
This file includes base classes for materials and matrial database class.
*/

#include <string>
#include <config.h>
#include <map>
#include <vector>
#include "../exceptions.h"

namespace plask {

//TODO lepiej użyć inteligętnych wskaźników czy obciążyć użytkowników potrzebą kasowania niektórych materiałów
//co jeśli klasa bazy materiałowej zostanie skasowana?

/**
 * Represent material, its physical properties.
 */
struct Material {

    ///Do nothing.
    virtual ~Material() {}

    // @return @c true only for dyanimicaly created materials which are not managed by MaterialDB and must be manualy deleted in some situations
    // virtual bool isDynamic() { return false; }

    ///@return material name
    virtual std::string getName() const = 0;

};

/**
 * Material which consist with few real materials.
 * It calculate avarages for all properties.
 */
struct MixedMaterial: public Material {

    //std::map<shared_ptr<Material>, double> materials;

    //virtual bool isDynamic() { return true; }

};

/**
 * Material which wrap one material and rotate its tensors properties.
 */
struct RotatedMaterial: public Material {

    shared_ptr<Material> wrapped;

};

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopand.
 *
 */
struct MaterialsDB {

    ///Amounts of dopands.
    enum DOPANT_AMOUNT_TYPE {
        NO_DOPANT,
        DOPING_CONCENTRATION,
        CARRIER_CONCENTRATION
    };

    typedef Material* construct_material_f(const std::vector<double>& components_amounts, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount);

private:
    std::map<std::string, construct_material_f*> constructors;

public:

    /**
     * Create material object.
     * @param parsed_name_with_donor material name with donor name in format material_name[:donor_name], for example: "AlGaN" or "AlGaN:Mg"
     * @param composition amounts of elements, with NaN for each element for composition was not writen
     * @param dopant_amount_type type of amount of dopand, needed to interpetation of @a dopant_amount
     * @param dopant_amount amount of dopand, is ignored if @a dopant_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @a parsed_name_with_donor
     */
    shared_ptr<Material> get(const std::string& parsed_name_with_donor, const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type = NO_DOPANT, double dopant_amount = 0.0) const throw (NoSuchMaterial);

    /**
     * Create material object.
     * @param name_with_components elements composition in format element1(amount1)...elementN(amountN), where some amounts are optional for example: "Al(0.7)GaN"
     * @param dopant_descr empty string if there is no dopand or description of dopant in format elementname=amount or elementname p/n=amount, for example: "Mg=7e18" or "Mg p=7e18"
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @a parsed_name_with_donor
     * @throw MaterialParseException if can't parse @a name_with_components or @a dopant_descr
     */
    shared_ptr<Material> get(const std::string& name_with_components, const std::string& dopant_descr) const throw (NoSuchMaterial, MaterialParseException);

    /**
     * Create material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see @ref get(const std::string& name_with_components, const std::string& dopant_descr)
     * @return material with @a full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @a full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const throw (NoSuchMaterial, MaterialParseException);

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param name material name (with donor after ':')
     * @param constructor function which can create material instance
     */
    void add(const std::string& name, construct_material_f* constructor);

    /**
     * Fill database with default materials creators.
     */
    //TODO materials will be cr
    //void init();
};

} // namespace plask

#endif	//PLASK__MATERIAL_H
