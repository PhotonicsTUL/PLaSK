#ifndef PLASK__MATERIAL_DB_H
#define PLASK__MATERIAL_DB_H

#include "material.h"

#include <boost/iterator/transform_iterator.hpp>

namespace plask {

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopant.
 */
struct MaterialsDB {

    /**
     * Object of this class (inharited from it) construct material instance.
     */
    struct MaterialConstructor {

        /**
         * Name of material which this constructor can create.
         */
        std::string materialName;

        /**
         * MaterialConstructor constructor.
         * @param materialName name of material which this constructor can create
         */
        MaterialConstructor(const std::string& materialName): materialName(materialName) {}

        /**
         * Create material.
         * @param composition parsed amounts of elements, can be not completed (see Material::completeComposition), empty composition in case of simple materials
         * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
         * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPANT
         */
        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double dopant_amount) const = 0;
    };

private:

    typedef std::map<std::string, shared_ptr<const MaterialConstructor> > constructors_map_t;

    /// Map: material db key -> materials constructors object
    //  (it needs to be public to enable access from Python interface)
    constructors_map_t constructors;

    struct iter_val: public std::unary_function<const constructors_map_t::value_type&, const constructors_map_t::mapped_type&> {
        const constructors_map_t::mapped_type& operator()(const constructors_map_t::value_type &pair) const { return pair.second; }
    };

public:

    typedef boost::transform_iterator<iter_val, constructors_map_t::const_iterator> iterator;
    typedef iterator const_iterator;

    const_iterator begin() const { return iterator(constructors.begin()); }
    const_iterator end() const { return iterator(constructors.end()); }

    /**
     * Throw excpetion if given composition is empty.
     * @param composition composition to check
     */
    static void ensureCompositionIsNotEmpty(const Material::Composition& composition);

    /**
     * Specialization of this implements MaterialConstructor.
     * operator() delegates call to Material constructor, eventualy ignoring (depending from requireComposition and requireDopant) some arguments.
     * @tparam MaterialType type of material
     * @tparam requireComposition if @c true ensure if comosition is not empty, material composition will be completed and passed to constructor,
     *                              if @c false composition will be ignored
     * @tparam requireDopant if @c true dopant information will be passed to constructor, if @c false dopant information will be ignored
     */
    template <typename MaterialType, bool requireComposition, bool requireDopant>
    struct DelegateMaterialConstructor;

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, true, true>: public MaterialConstructor {
        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}
        virtual shared_ptr<MaterialType> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const {
            ensureCompositionIsNotEmpty(composition);
            return shared_ptr<Material>(new MaterialType(Material::completeComposition(composition), doping_amount_type, doping_amount));
        }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, true, false>: public MaterialConstructor {
        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}
        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE, double) const {
            ensureCompositionIsNotEmpty(composition);
            return shared_ptr<Material>(new MaterialType(Material::completeComposition(composition)));
        }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, false, true>: public MaterialConstructor {
        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}
        virtual shared_ptr<Material> operator()(const Material::Composition&, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const {
            return shared_ptr<Material>(new MaterialType(doping_amount_type, doping_amount));
        }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, false, false>: public MaterialConstructor {
        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}
        virtual shared_ptr<Material> operator()(const Material::Composition&, Material::DOPING_AMOUNT_TYPE, double) const {
            return shared_ptr<Material>(new MaterialType());
        }
    };

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
     * Add simple material (which not require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance; must be created by new operator and material DB will call delete for it
     */
    void addSimple(const MaterialConstructor* constructor);
    
    /**
     * Add complex material (which require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance; must be created by new operator and material DB will call delete for it
     */
    void addComplex(const MaterialConstructor* constructor);

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     *
     * Use DelegateMaterialConstructor as material construction object.
     * @param name material name (with dopant after ':')
     * @tparam MaterialType, requireComposition, requireDopant see DelegateMaterialConstructor
     */
    template <typename MaterialType, bool requireComposition, bool requireDopant>
    void add(const std::string& name) {
        if (requireComposition)
            addComplex(new DelegateMaterialConstructor<MaterialType, requireComposition, requireDopant>(name));
        else
            addSimple(new DelegateMaterialConstructor<MaterialType, requireComposition, requireDopant>(name));
    }

    template <typename MaterialType>
    void add(const std::string& name) {
        add<MaterialType, MaterialType::USE_COMPOSITION, MaterialType::HAS_DOPANT>(name);
    }

    /**
     * Fill database with default materials creators.
     */
    //TODO materials will be created
    //void init();

private:

    /**
     * Create material object.
     * @param dbKey key in database
     * @param composition elements composition, empty composition for simple materials
     * @param dopant_name name of dopant (if any)
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if there is no material with key @p dbKey in database
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const std::string& dbKey, const Material::Composition& composition,
                             const std::string& dopant_name = "", Material::DOPING_AMOUNT_TYPE doping_amount_type = Material::NO_DOPING, double doping_amount = 0.0) const;

    
};

}   // namespace plask

#endif // PLASK__MATERIAL_DB_H
