/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__MATERIAL_DB_H
#define PLASK__MATERIAL_DB_H

#include <functional>
#include "material.hpp"
#include "const_material.hpp"

#include <boost/iterator/transform_iterator.hpp>
#include "../utils/system.hpp"
#include "../utils/warnings.hpp"

#include "info.hpp"

namespace plask {

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopant.
 */
struct PLASK_API MaterialsDB {

    /**
     * Get default material database.
     * @return default material database
     */
    static MaterialsDB& getDefault();

    /// Replace default database with temporary (or empty) value and revert its original state in destructor.
    struct TemporaryReplaceDefault;
    typedef TemporaryReplaceDefault TemporaryClearDefault;

    /**
     * Load materials from file (dynamic library which is in plask directory with materials) to default database.
     * @param fileName_mainpart main part of file name (filename without lib prefix and extension)
     */
    static void loadToDefault(const std::string& fileName_mainpart);

    /**
     * Load all materials from given @p dir (plask materials directory by default) to default database.
     * @param dir directory with materials shared libraries (all files from this dir will be loaded)
     */
    static void loadAllToDefault(const std::string& dir = plaskMaterialsPath());

    /**
     * Clear the database
     */
    void clear() {
        constructors.clear();
        info.clear();
    }

    /**
     * Update with values from different database
     * \param src source database
     */
    void update(const MaterialsDB& src) {
        for (const auto& item: src.constructors) {
            constructors[item.first] = item.second;
        }
        info.update(src.info);
    }

    /**
     * Helper which calls getDefault().add<MaterialType>([name]) in constructor.
     *
     * Creating global objects of this type allow to fill the default database.
     */
    template <typename MaterialType>
    struct Register {
        Register(const std::string& name) { getDefault().add<MaterialType>(name); }
        Register() { getDefault().add<MaterialType>(); }
    };

    /// Same as Register but for materials without static field NAME.
    template <typename MaterialType>
    struct RegisterUnderName {
        RegisterUnderName(const std::string& name) { getDefault().add<MaterialType>(name); }
    };

    /**
     * Object of this class (inherited from it) construct material instance.
     * It produces materials of one type but with various composition and ammount of dopant.
     */
    struct PLASK_API MaterialConstructor {

        /**
         * Full name (with optional dopant name) of material which this constructor can create.
         */
        std::string materialName;

        /**
         * MaterialConstructor constructor.
         * @param materialName full name (with optional dopant name) of material which this constructor can create
         */
        MaterialConstructor(const std::string& materialName): materialName(materialName) {}

        /**
         * Create material.
         * @param composition parsed amounts of objects, can be not completed (see Material::completeComposition), empty composition in case of simple materials
         * @param doping amount of dopant
         * @return created material
         */
        virtual shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const = 0;

        /**
         * @return @c true only if this constructor creates simple material (does not use composition)
         */
        virtual bool isAlloy() const = 0;

        virtual ~MaterialConstructor() {}

        void ensureCompositionIsEmpty(const Material::Composition& composition) const {
            if (!composition.empty()) throw Exception("redundant composition given for material '{0}'", materialName);
        }

        void ensureNoDoping(double doping) const {
            if (!isnan(doping) && doping != 0.) throw Exception("redundant doping given for material '{0}'", materialName);
        }
    };

    /**
     * Base class for factories of alloy materials which construct their versions with mixed compositions and/or doping amounts.
     */
    struct PLASK_API MixedCompositionFactory {

      protected:

        shared_ptr<const MaterialConstructor> constructor;

      public:
        /**
         * Construct MixedCompositionFactory for given material constructor and two compositions for this constructor.
         * @param constructor material constructor
         */
        MixedCompositionFactory(shared_ptr<const MaterialConstructor> constructor): constructor(constructor) {}

        virtual ~MixedCompositionFactory() {}

        /**
         * Construct material.
         * @param m1_weight weight of first composition or doping amount
         * @return constructed material
         */
        virtual shared_ptr<Material> operator()(double m1_weight) const = 0;

        /**
         * Get material only if this factory represents solid material (if operator(double m1_weight) is independent from m1_weight).
         * @return material or nullptr if it is not solid
         */
        virtual shared_ptr<Material> singleMaterial() const = 0;

    };

    /**
     * Factory of alloy material which construct it version with mixed version of two compositions (for materials without dopants).
     */
    //TODO cache: double -> constructed material
    struct PLASK_API MixedCompositionOnlyFactory: public MixedCompositionFactory {    //TODO nonlinear mixing with functor double [0.0, 1.0] -> double [0.0, 1.0]

      protected:

        Material::Composition material1composition, material2composition;

        double shape;

        /**
         * Calculate mixed composition, of material1composition and material2composition.
         * @param m1_weight weight of first composition (material1composition)
         * @return incomplate, mixed composision
         */
        Material::Composition mixedComposition(double m1_weight) const;

      public:
        /**
         * Construct MixedCompositionFactory for given material constructor and two compositions for this constructor.
         * @param constructor material constructor
         * @param material1composition incomplate composition of first material
         * @param material2composition incomplate composition of second material, must be defined for the same objects as @p material1composition
         * \param shape changing material shape exponent
         */
        MixedCompositionOnlyFactory(shared_ptr<const MaterialConstructor> constructor, const Material::Composition& material1composition, const Material::Composition& material2composition, double shape=1.);

        /**
         * Construct material.
         * @param m1_weight weight of first composition
         * @return constructed material
         */
        shared_ptr<Material> operator()(double m1_weight) const override {
            return (*constructor)(mixedComposition(m1_weight), NAN);
        }

        shared_ptr<Material> singleMaterial() const override {
            return material1composition == material2composition ? (*constructor)(material1composition, NAN) : shared_ptr<Material>();
        }
    };

    /**
     * Factory of alloy material which construct its versions with mixed version of two compositions and dopants.
     */
    struct PLASK_API MixedCompositionAndDopantFactory: public MixedCompositionOnlyFactory {
      protected:
        double m1DopAmount, m2DopAmount;

      public:
        /**
         * Construct MixedCompositionAndDopantFactory for given material constructor, two compositions and dopings amounts for this constructor.
         * @param constructor material constructor
         * @param material1composition incomplate composition of first material
         * @param material2composition incomplate composition of second material, must be defined for the same objects as @p material1composition
         * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
         * \param shape changing material shape exponent
         */
        MixedCompositionAndDopantFactory(shared_ptr<const MaterialConstructor> constructor, const Material::Composition& material1composition, const Material::Composition& material2composition,
                                         double m1DopAmount, double m2DopAmount, double shape=1.)
            : MixedCompositionOnlyFactory(constructor, material1composition, material2composition, shape), m1DopAmount(m1DopAmount), m2DopAmount(m2DopAmount) {}

        /**
         * Construct material.
         * @param m1_weight weight of first composition and dopant
         * @return constructed material
         */
        shared_ptr<Material> operator()(double m1_weight) const override {
            return (*constructor)(mixedComposition(m1_weight),
                                  m1DopAmount * pow(m1_weight, shape) + m2DopAmount * (1.0 - pow(m1_weight, shape)));
        }

        shared_ptr<Material> singleMaterial() const override {
            return (material1composition == material2composition) && (m1DopAmount == m2DopAmount) ?
                        (*constructor)(material1composition, m1DopAmount) : shared_ptr<Material>();
        }
    };

    /**
     * Factory of alloy material which construct its versions with mixed versions of two dopants (for material with same compositions).
     */
    struct PLASK_API MixedDopantFactory: public MixedCompositionFactory {
      protected:
        double m1DopAmount, m2DopAmount;

        double shape;

      public:
        /**
         * Construct MixedDopantFactory for given material constructor of simple material, and doping amounts for this constructor.
         * @param constructor material constructor
         * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
         * \param shape changing material shape exponent
         */
        MixedDopantFactory(shared_ptr<const MaterialConstructor> constructor, double m1DopAmount, double m2DopAmount, double shape=1.)
            : MixedCompositionFactory(constructor), m1DopAmount(m1DopAmount), m2DopAmount(m2DopAmount), shape(shape) {}

        /**
         * Construct material.
         * @param m1_weight weight of first composition and dopant
         * @return constructed material
         */
        shared_ptr<Material> operator()(double m1_weight) const override {
            return (*constructor)(Material::Composition(), m1DopAmount * pow(m1_weight, shape) + m2DopAmount * (1.0 - pow(m1_weight, shape)));
        }

        shared_ptr<Material> singleMaterial() const override {
            return m1DopAmount == m2DopAmount ? (*constructor)(Material::Composition(), m1DopAmount) : shared_ptr<Material>();
        }
    };

    /**
     * Dummy mixed factory for use in draft mode
     */
    struct PLASK_API DummyMixedCompositionFactory: public MixedCompositionFactory {
      protected:
        std::string full_name;

      public:
        /**
         * Construct MixedDopantFactory for given material constructor of simple material, and doping amounts for this constructor.
         * @param constructor material constructor
         * @param dopAmountType type of doping amounts, common for both materials
         * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
         * \param shape changing material shape exponent
         */
        DummyMixedCompositionFactory(const std::string& name1, const std::string& name2)
            : MixedCompositionFactory(shared_ptr<const MaterialConstructor>()), full_name(name1 + "..." + name2) {}

        /**
         * Construct material.
         * @param m1_weight weight of first composition and dopant
         * @return constructed material
         */
        shared_ptr<Material> operator()(double PLASK_UNUSED(m1_weight)) const override {
            return plask::make_shared<DummyMaterial>(full_name);
        }

        shared_ptr<Material> singleMaterial() const override {
            return plask::make_shared<DummyMaterial>(full_name);
        }
    };

  private:

    /// Type for map: material db key -> materials constructors object (with name)
    typedef std::map<std::string, shared_ptr<const MaterialConstructor> > constructors_map_t;

    /// Map: material db key -> materials constructors object (with name)
    constructors_map_t constructors;

    // Static const constructors_map_t::mapped_type& iter_val(const constructors_map_t::value_type &pair) { return pair.second; }
    struct iter_val: public std::function<const constructors_map_t::mapped_type&(const constructors_map_t::value_type&)> {
        const constructors_map_t::mapped_type& operator()(const constructors_map_t::value_type &pair) const { return pair.second; }
    };

public:

    /// Info for this database
    MaterialInfo::DB info;

    /// Iterator over material constructors (shared_ptr<shared_ptr<const MaterialConstructor>>).
    typedef boost::transform_iterator<iter_val, constructors_map_t::const_iterator> iterator;

    /// Iterator over material constructors (shared_ptr<shared_ptr<const MaterialConstructor>>).
    typedef iterator const_iterator;

    /**
     * Get iterator which refer to first record in database.
     * @return iterator which refer to first record in database
     */
    const_iterator begin() const { return iterator(constructors.begin()); }

    /**
     * Get iterator which refer one step after last record in database.
     * @return iterator which refer just after last record in database
     */
    const_iterator end() const { return iterator(constructors.end()); }

    /**
     * Get number of material types in database.
     *
     * This is not the same as number of materials in database,
     * because one material type can have many materials with different compositions and dopants.
     * \return number of material types in database
     */
    size_t size() const {
        return constructors.size();
    }

    /**
     * Throw exception if given composition is empty.
     * @param composition composition to check
     */
    static void ensureCompositionIsNotEmpty(const Material::Composition& composition);

    /**
     * Specialization of this implements MaterialConstructor.
     *
     * operator() delegates call to Material constructor, optionally ignoring (depending from requireComposition and requireDopant) some arguments.
     * @tparam MaterialType type of material
     * @tparam requireComposition if @c true ensure if comosition is not empty, material composition will be completed and passed to constructor,
     *                              if @c false composition will be ignored
     * @tparam requireDopant if @c true dopant information will be passed to constructor, if @c false dopant information will be ignored
     */
    template <typename MaterialType,
              bool requireComposition = Material::is_with_composition<MaterialType>::value,
              bool requireDopant = Material::is_with_dopant<MaterialType>::value >
    struct DelegateMaterialConstructor;

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, true, true>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override {
            ensureCompositionIsNotEmpty(composition);
            return plask::make_shared<MaterialType>(Material::completeComposition(composition), doping);
        }

        bool isAlloy() const override { return true; }    // == ! requireComposition
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, true, false>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override {
            ensureCompositionIsNotEmpty(composition);
            ensureNoDoping(doping);
            return plask::make_shared<MaterialType>(Material::completeComposition(composition));
        }

        bool isAlloy() const override { return true; }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, false, true>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override {
            ensureCompositionIsEmpty(composition);
            return plask::make_shared<MaterialType>(doping);
        }

        bool isAlloy() const override { return false; }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, false, false>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override {
            ensureCompositionIsEmpty(composition);
            ensureNoDoping(doping);
            return plask::make_shared<MaterialType>();
        }

        bool isAlloy() const override { return false; }
    };

    /**
     * Material constructor that holds other constructor or complete material object based on the provided name.
     */
    class PLASK_API ProxyMaterialConstructor: public MaterialConstructor {
        shared_ptr<Material> material;
        shared_ptr<const MaterialsDB::MaterialConstructor> constructor;
        Material::Composition composition;
        double doping;

      public:
        ProxyMaterialConstructor();

        ProxyMaterialConstructor(const std::string& name, const MaterialsDB& db=MaterialsDB::getDefault());

        ProxyMaterialConstructor(const shared_ptr<Material>& material);

        shared_ptr<Material> operator()(const Material::Composition& comp, double dop) const override;

        bool isAlloy() const override;
    };

    /**
     * Create material object.
     * @param composition complete objects composition
     * @param label optional material label
     * @param dopant_name name of dopant
     * @param doping amount of dopant
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const Material::Composition& composition, const std::string& label, const std::string& dopant_name, double doping) const;

    /**
     * Create material object.
     * @param composition complete objects composition
     * @param dopant_name name of dopant
     * @param doping amount of dopant
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const Material::Composition& composition, const std::string& dopant_name, double doping) const {
        return get(composition, "", dopant_name, doping);
    }

    /**
     * Create material object.
     * @param composition complete objects composition
     * @param label optional material label
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const Material::Composition& composition, const std::string& label = "") const {
        return get(composition, label, "", 0.);
    }

    /**
     * Create material object.
     * @param name_with_dopant material name with dopant name in format material[_label][:dopant_name], for example: "Al(0.2)GaN" or "Al(0.2)GaN:Mg"
     * @param doping amount of dopant
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     */
    shared_ptr<Material> get(const std::string& name_with_dopant, double doping) const;

    /*
     * Create material object.
     * @param name_with_components objects composition in format object1(amount1)...objectN(amountN), where some amounts are optional for example: "Al(0.7)GaN"
     * @param doping_descr empty string if there is no doping or description of dopant in format objectname=amount or objectname p/n=amount, for example: "Mg=7e18" or "Mg p=7e18"
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @throw MaterialParseException if can't parse @p name_with_components or @p doping_descr
     */
    //shared_ptr<Material> get(const std::string& name_with_components, const std::string& doping_descr) const;

    /**
     * Get constructor of material.
     * @param name_without_composition material name, without encoded parameters, in format composition[_label][:dopant]
     * @return constructor of material or nullptr if there is no such material in database
     * @throw NoSuchMaterial if database doesn't know material with name @p name_without_composition
     */
    shared_ptr<const MaterialConstructor> getConstructor(const std::string& name_without_composition) const;

    shared_ptr<Material> get(const Material::Parameters& param) const;

    /**
     * Create material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see get(const std::string&, const std::string&)
     * @return material with @p full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @p full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const;

    /**
     * Construct mixed material factory.
     * @param material1_fullname, material2_fullname materials name, with encoded parameters in format composition[_label][:dopant],
     *      both must refer to the same material with the same dopant and in case of doping materials, amounts of dopants must be given in the same format
     * \param shape changing material shape exponent
     * @return constructed factory
     */
    shared_ptr<MixedCompositionFactory> getFactory(const std::string& material1_fullname, const std::string& material2_fullname, double shape=1.) const;

    /*
     * Add simple material (which does snot require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance; must be created by operator new and material DB will call delete for it
     */
   // void addSimple(const MaterialConstructor* constructor);

    /**
     * Add simple material (which does snot require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance
     */
    void addSimple(shared_ptr<MaterialConstructor> constructor);

    /*
     * Add alloy material (which require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance; must be created by operator new and material DB will call delete for it
     */
    //void addAlloy(const MaterialConstructor* constructor);

    /**
     * Add alloy material (which require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance
     */
    void addAlloy(shared_ptr<MaterialConstructor> constructor);

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
            addAlloy(plask::make_shared< DelegateMaterialConstructor<MaterialType, requireComposition, requireDopant> >(name));
        else
            addSimple(plask::make_shared< DelegateMaterialConstructor<MaterialType, requireComposition, requireDopant> >(name));
    }

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     *
     * Use DelegateMaterialConstructor as material construction object.
     * Deduce from constructors if material needs either composition or dopant information.
     * @param name material name (with dopant after ':')
     */
    template <typename MaterialType>
    void add(const std::string& name) {
        add<MaterialType, Material::is_with_composition<MaterialType>::value, Material::is_with_dopant<MaterialType>::value>(name);
    }

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     *
     * Use DelegateMaterialConstructor as material construction object.
     * Deduce from constructors if material needs either composition or dopant information.
     * Material name is read from static field MaterialType::NAME.
     */
    template <typename MaterialType>
    void add() { add<MaterialType>(MaterialType::NAME); }

    /**
     * Remove material from DB.
     *
     * Deduce from name if material is either alloy or simple.
     * @param name material name (with dopant after ':')
     */
    void remove(const std::string& name);

    /**
     * Check if a material (given without parameters) is simple.
     * @param name_without_composition material name, without encoded parameters, in format composition[:dopant]
     * @return @c true only if the material is simple
     * @throw NoSuchMaterial if database doesn't know the material with name @p name_without_composition
     */
    bool isAlloy(const std::string& material_name) const;

    /*
     * Get alloy material constructor object.
     * @param composition objects composition
     * @param dopant_name name of dopant (if any)
     */
    shared_ptr<const MaterialConstructor> getConstructor(const Material::Composition& composition, const std::string& label = "", const std::string& dopant_name = "") const;

    shared_ptr<const MaterialConstructor> getConstructor(const Material::Parameters& material, bool allow_alloy_without_composition = false) const;

private:

    /**
     * Get material constructor object.
     * @param dbKey key in database (format: name[_label] or normalized_composition[_label])
     * @param composition objects composition, empty composition for simple materials, used for error checking and messages
     * @param allow_alloy_without_composition if true alloy material can be obtained if composition is empty (if false exception will be thrown in such situation when dbKey is not simple material)
     */
    shared_ptr<const MaterialConstructor> getConstructor(const std::string& dbKey, const Material::Composition& composition, bool allow_alloy_without_composition = false) const;

    /**
     * Create material object.
     * @param dbKey key in database
     * @param composition objects composition, empty composition for simple materials
     * @param doping amount of dopant
     * @return constructed material
     * @throw NoSuchMaterial if there is no material with key @p dbKey in database
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const std::string& dbKey, const Material::Composition& composition, double doping = 0.0) const;

};

struct PLASK_API MaterialsDB::TemporaryReplaceDefault {
    MaterialsDB toRevert;

    TemporaryReplaceDefault(const TemporaryReplaceDefault&) = delete;
    TemporaryReplaceDefault& operator=(const TemporaryReplaceDefault&) = delete;
    TemporaryReplaceDefault(TemporaryReplaceDefault&&) = delete;
    TemporaryReplaceDefault& operator=(TemporaryReplaceDefault&&) = delete;

    /**
     * Construct an object which replace default database with @p temporaryValue and revert its original state in destructor.
     * @param temporaryValue new, temporary value for default database (the empty database by default)
     */
    TemporaryReplaceDefault(MaterialsDB temporaryValue = MaterialsDB()): toRevert(std::move(getDefault())) {
        getDefault() = std::move(temporaryValue);
    }

    ~TemporaryReplaceDefault() {
        MaterialsDB::getDefault() = std::move(toRevert);
    }
};

}   // namespace plask

#endif // PLASK__MATERIAL_DB_H
