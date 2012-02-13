#ifndef PLASK__MATERIAL_DB_H
#define PLASK__MATERIAL_DB_H

#include "material.h"

namespace plask {

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopant.
 *
 */
struct MaterialsDB {

    /**
     * Object of this class (inharited from it) construct material instance.
     */
    struct MaterialConstructor {

        /**
         * Create material.
         * @param composition amounts of elements (completed)
         * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
         * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPANT
         */
        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double dopant_amount) const = 0;

    };

    /**
     * Type of function which construct material.
     * @param composition amounts of elements (completed)
     * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPANT
     */
    typedef Material* construct_material_f(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double dopant_amount);

    /**
     * Construct material instance using construct_f function.
     */
    struct FunctionBasedMaterialConstructor: public MaterialConstructor {
        construct_material_f* constructFunction;
        FunctionBasedMaterialConstructor(construct_material_f* constructFunction): constructFunction(constructFunction) {}
        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double dopant_amount) const {
            return shared_ptr<Material>(constructFunction(composition, doping_amount_type, dopant_amount));
        }
    };


    /**
     * Template of function which construct material (which require information about composition and dopant) with given type.
     * @param composition amounts of elements (completed)
     * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPING
     * @tparam MaterialType type of material to construct, must fill requirements:
     * - inherited from plask::Material
     * - must have constructor which takes parameters: Material::Composition composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount
     * - this constructor can suppose that composition is complete (defined for all element which material includes and without NaN)
     * @see construct_comp, construct_dop, construct
     */
    template <typename MaterialType> Material* construct_comp_dop(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) {
        return new MaterialType(composition, doping_amount_type, doping_amount);
    }
    
    /**
     * Template of function which construct material (which require information about composition) with given type.
     * @param composition amounts of elements (completed)
     * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPING
     * @tparam MaterialType type of material to construct, must fill requirements:
     * - inherited from plask::Material
     * - must have constructor which takes parameters: Material::Composition composition
     * - this constructor can suppose that composition is complete (defined for all element which material includes and without NaN)
     * @see construct_comp_dop, construct_dop, construct
     */
    template <typename MaterialType> Material* construct_comp(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) {
        return new MaterialType(composition);
    }
    
    /**
     * Template of function which construct material (which require information about dopant) with given type.
     * @param composition amounts of elements (completed)
     * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPING
     * @tparam MaterialType type of material to construct, must fill requirements:
     * - inherited from plask::Material
     * - must have constructor which takes parameters: Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount
     * @see construct_comp_dop, construct_comp, construct
     */
    template <typename MaterialType> Material* construct_dop(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) {
        return new MaterialType(doping_amount_type, doping_amount);
    }
    
    /**
     * Template of function which construct material with given type.
     * @param composition amounts of elements (completed)
     * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPING
     * @tparam MaterialType type of material to construct, must fill requirements:
     * - inherited from plask::Material
     * - must have constructor which takes no parameters
     * @see construct_comp_dop, construct_comp, construct_dop
     */
    template <typename MaterialType> Material* construct(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) {
        return new MaterialType();
    }

    /// Map: material name -> materials constructors functions
    //  (it needs to be public to enable access from Python interface)
    std::map<std::string, std::unique_ptr<const MaterialConstructor> > constructors;

    /**
     * Create material object.
     * @param composition complete elements composition
     * @param dopant_name name of dopant (if any)
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const Material::Composition& composition, const std::string& dopant_name = "", Material::DOPING_AMOUNT_TYPE doping_amount_type = Material::NO_DOPING, double doping_amount = 0.0) const;
    
    /**
     * Create material object.
     * @param parsed_name_with_dopant material name with dopant name in format material_name[:dopant_name], for example: "AlGaN" or "AlGaN:Mg"
     * @param composition amounts of elements, with NaN for each element for composition was not written
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     */
    shared_ptr<Material> get(const std::string& parsed_name_with_dopant, const std::vector<double>& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type = Material::NO_DOPING, double doping_amount = 0.0) const;

    /**
     * Create material object.
     * @param name_with_components elements composition in format element1(amount1)...elementN(amountN), where some amounts are optional for example: "Al(0.7)GaN"
     * @param doping_descr empty string if there is no doping or description of dopant in format elementname=amount or elementname p/n=amount, for example: "Mg=7e18" or "Mg p=7e18"
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @throw MaterialParseException if can't parse @p name_with_components or @p doping_descr
     */
    shared_ptr<Material> get(const std::string& name_with_components, const std::string& doping_descr) const;

    /**
     * Create material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see @ref get(const std::string& name_with_components, const std::string& dopant_descr)
     * @return material with @p full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @p full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const;

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param elemenNames names of elements in material composition
     * @param dopant dopant name (empty if no dopant)
     * @param constructor object which can create material instance; must be created by new operator and material DB will call delete for it
     */
    void add(std::vector<std::string> elemenNames, const std::string& dopant, const MaterialConstructor* constructor);
    
    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param name material name (with dopant after ':')
     * @param constructor object which can create material instance; must be created by new operator and material DB will call delete for it
     */
    void add(const std::string& name, const MaterialConstructor* constructor);

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param name material name (with dopant after ':')
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
    //TODO materials will be created
    //void init();
    
};

}   // namespace plask

#endif // PLASK__MATERIAL_DB_H
