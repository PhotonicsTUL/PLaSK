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

/**
 * Represent material, its physical properties.
 */
struct Material {

    /// Do nothing.
    virtual ~Material() {}

    /// @return material name
    virtual std::string getName() const = 0;

};

/**
 * Material which consist with few real materials.
 * It calculate avarages for all properties.
 */
struct MixedMaterial: public Material {

    //std::map<shared_ptr<Material>, double> materials;

};

/**
 * Material which wrap one material and rotate its tensors properties.
 */
struct RotatedMaterial: public Material {

    shared_ptr<Material> wrapped;

};

/**
 * Check if material composition is compatibile with pattern and change NaN-s in composition to calculated ammounts.
 * @param composition ammounts of elements composition with NaN on position for which ammounts has not been taken
 * @param pattern sizes of elements groups
 * @return version of @a composition complement with calculated ammounts
 */
std::vector<double> fillMaterialCompositionAmounts(const std::vector<double>& composition, unsigned pattern);

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopand.
 *
 */
struct MaterialsDB {

    ///Amounts of dopands.
    enum DOPANT_AMOUNT_TYPE {
        NO_DOPANT,              ///< no dopand
        DOPING_CONCENTRATION,   ///< doping concentration
        CARRIER_CONCENTRATION   ///< carrier concentration
    };

    typedef Material* construct_material_f(const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount);

    /**
     * Construct material with given type.
     * @tparam MaterialType type of material to construct, must fill requirments:
     * - inharited from plask::Material
     * - has public, static unsigned COMPOSITION_PATTERN field which determinates sizes of composition groups (for example: 21 means that there are two groups, first group has size 2 and second has size 1)
     * - must have constructor which takes parameters: std::vector<double> composition, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount
     */
    //TODO set some by methods? what with materials witout dopands?
    template <typename MaterialType> Material* construct(const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) {
        return new MaterialType( fillMaterialCompositionAmounts(MaterialType::COMPOSITION_PATTERN), dopant_amount_type, dopant_amount );
    }

private:
    ///Map: material name -> materials constructors functions
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
    shared_ptr<Material> get(const std::string& parsed_name_with_donor, const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type = NO_DOPANT, double dopant_amount = 0.0) const;

    /**
     * Create material object.
     * @param name_with_components elements composition in format element1(amount1)...elementN(amountN), where some amounts are optional for example: "Al(0.7)GaN"
     * @param dopant_descr empty string if there is no dopand or description of dopant in format elementname=amount or elementname p/n=amount, for example: "Mg=7e18" or "Mg p=7e18"
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @a parsed_name_with_donor
     * @throw MaterialParseException if can't parse @a name_with_components or @a dopant_descr
     */
    shared_ptr<Material> get(const std::string& name_with_components, const std::string& dopant_descr) const;

    /**
     * Create material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see @ref get(const std::string& name_with_components, const std::string& dopant_descr)
     * @return material with @a full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @a full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const;

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param name material name (with donor after ':')
     * @param constructor function which can create material instance
     */
    void add(const std::string& name, construct_material_f* constructor);

    /**
     * Add material with given type to DB.
     * All information about material (like name, composition amount pattern) are read from static MaterialType fields.
     */
    //template <typename MaterialType> add();

    /**
     * Fill database with default materials creators.
     */
    //TODO materials will be cr
    //void init();
};

} // namespace plask

#endif	//PLASK__MATERIAL_H
