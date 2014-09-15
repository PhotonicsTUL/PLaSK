#ifndef PLASK__MATERIAL_DB_H
#define PLASK__MATERIAL_DB_H

#include <functional>
#include "material.h"

#include <boost/iterator/transform_iterator.hpp>
#include "../utils/system.h"

namespace plask {

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopant.
 */
struct PLASK_API MaterialsDB {

    /*
     * Materials source functor which use materials database.
     */
    /*struct Source {
        const MaterialsDB& materialsDB;
        Source(const MaterialsDB& materialsDB): materialsDB(materialsDB) {}
        shared_ptr<Material> operator()(const std::string& material_full_name) const { return materialsDB.get(material_full_name); }
    };*/

    /*
     * Get material database wrapped by @p materialsSource.
     * @param materialsSource source of materials, which can wrap material database
     * @return pointer to wrapped material database or @c nullptr if materialsSource is not constructed using material database
     */
    //static const MaterialsDB* getFromSource(const MaterialsSource& materialsSource);

    /*
     * Convert @c this material database to material source.
     * @return material source which use @c this database
     */
    /*Source toSource() const {
        return Source(*this);
    }*/

    /**
     * Get default material database.
     * @return default material database
     */
    static MaterialsDB& getDefault();

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
     * It produces materials of one type but with variouse composition and ammount of dopant.
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
         * @param dopant_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
         * @param dopant_amount amount of dopant, is ignored if @p dopant_amount_type is @c NO_DOPANT
         * @return created material
         */
        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType dopant_amount_type, double dopant_amount) const = 0;

        /**
         * @return @c true only if this contructor creates simple material (does not use composition)
         */
        virtual bool isSimple() const = 0;

        virtual ~MaterialConstructor() {}

        static void ensureCompositionIsEmpty(const Material::Composition& composition) {
            if (!composition.empty()) throw Exception("Redundant composition given for material \"%1%\".");
        }

        static void ensureDopantIsNo(Material::DopingAmountType dopant_amount_type) {
            if (dopant_amount_type != Material::NO_DOPING) throw Exception("Redundant dopant given for material \"%1%\".");
        }
    };

    /**
     * Base class for factories of complex material which construct it version with mixed version of two compositions and/or doping amounts.
     */
    struct PLASK_API MixedCompositionFactory {    //TODO mieszanie nie liniowe, funkcja, funktor double [0.0, 1.0] -> double [0.0, 1.0]

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
         * Get material only if it this factory represents solid material (if operator(double m1_weight) is independent from m1_weight).
         * @return material or nullptr if it is not solid
         */
        virtual shared_ptr<Material> singleMaterial() const = 0;

    };


    /**
     * Factory of complex material which construct it version with mixed version of two compositions (for materials without dopants).
     */
    //TODO cache: double -> constructed material
    struct PLASK_API MixedCompositionOnlyFactory: public MixedCompositionFactory {    //TODO mieszanie nieliniowe, funkcja, funktor double [0.0, 1.0] -> double

    protected:

        Material::Composition material1composition, material2composition;

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
         */
        MixedCompositionOnlyFactory(shared_ptr<const MaterialConstructor> constructor, const Material::Composition& material1composition, const Material::Composition& material2composition);

        /**
         * Construct material.
         * @param m1_weight weight of first composition
         * @return constructed material
         */
        shared_ptr<Material> operator()(double m1_weight) const {
            return (*constructor)(mixedComposition(m1_weight), Material::NO_DOPING, 0.0);
        }

        virtual shared_ptr<Material> singleMaterial() const {
            return material1composition == material2composition ? (*constructor)(material1composition, Material::NO_DOPING, 0.0) : shared_ptr<Material>();
        }
    };

    /**
     * Factory of complex material which construct it version with mixed version of two compositions and dopants.
     */
    struct PLASK_API MixedCompositionAndDopantFactory: public MixedCompositionOnlyFactory {
    protected:
        Material::DopingAmountType dopAmountType;

        double m1DopAmount, m2DopAmount;

    public:
        /**
         * Construct MixedCompositionAndDopantFactory for given material constructor, two compositions and dopings amounts for this constructor.
         * @param constructor material constructor
         * @param material1composition incomplate composition of first material
         * @param material2composition incomplate composition of second material, must be defined for the same objects as @p material1composition
         * @param dopAmountType type of doping amounts, common for @p m1DopAmount and @p m2DopAmount
         * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
         */
        MixedCompositionAndDopantFactory(shared_ptr<const MaterialConstructor> constructor, const Material::Composition& material1composition, const Material::Composition& material2composition,
                                         Material::DopingAmountType dopAmountType, double m1DopAmount, double m2DopAmount)
            : MixedCompositionOnlyFactory(constructor, material1composition, material2composition), dopAmountType(dopAmountType), m1DopAmount(m1DopAmount), m2DopAmount(m2DopAmount) {}

        /**
         * Construct material.
         * @param m1_weight weight of first composition and dopant
         * @return constructed material
         */
        shared_ptr<Material> operator()(double m1_weight) const {
            return (*constructor)(mixedComposition(m1_weight), dopAmountType,
                                  m1DopAmount * m1_weight + m2DopAmount * (1.0 - m1_weight));
        }

        virtual shared_ptr<Material> singleMaterial() const {
            return (material1composition == material2composition) && (m1DopAmount == m2DopAmount) ?
                        (*constructor)(material1composition, dopAmountType, m1DopAmount) : shared_ptr<Material>();
        }
    };

    /**
     * Factory of complex material which construct it version with mixed version of two dopants (for material with same compositions).
     */
    struct PLASK_API MixedDopantFactory: public MixedCompositionFactory {
    protected:
        Material::DopingAmountType dopAmountType;

        double m1DopAmount, m2DopAmount;

    public:
        /**
         * Construct MixedDopantFactory for given material constructor of simple material, and doping amounts for this constructor.
         * @param constructor material constructor
         * @param dopAmountType type of doping amounts, common for both materials
         * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
         */
        MixedDopantFactory(shared_ptr<const MaterialConstructor> constructor, Material::DopingAmountType dopAmountType, double m1DopAmount, double m2DopAmount)
            : MixedCompositionFactory(constructor), dopAmountType(dopAmountType), m1DopAmount(m1DopAmount), m2DopAmount(m2DopAmount) {}

        /**
         * Construct material.
         * @param m1_weight weight of first composition and dopant
         * @return constructed material
         */
        shared_ptr<Material> operator()(double m1_weight) const {
            return (*constructor)(Material::Composition(), dopAmountType, m1DopAmount * m1_weight + m2DopAmount * (1.0 - m1_weight));
        }

        virtual shared_ptr<Material> singleMaterial() const {
            return m1DopAmount == m2DopAmount ? (*constructor)(Material::Composition(), dopAmountType, m1DopAmount) : shared_ptr<Material>();
        }
    };



private:

    /// Type for map: material db key -> materials constructors object (with name)
    typedef std::map<std::string, shared_ptr<const MaterialConstructor> > constructors_map_t;

    /// Map: material db key -> materials constructors object (with name)
    constructors_map_t constructors;

    //static const constructors_map_t::mapped_type& iter_val(const constructors_map_t::value_type &pair) { return pair.second; }
    struct iter_val: public std::unary_function<const constructors_map_t::value_type&, const constructors_map_t::mapped_type&> {
        const constructors_map_t::mapped_type& operator()(const constructors_map_t::value_type &pair) const { return pair.second; }
    };

public:

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
     * Throw excpetion if given composition is empty.
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

        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const override {
            ensureCompositionIsNotEmpty(composition);
            return make_shared<MaterialType>(Material::completeComposition(composition), doping_amount_type, doping_amount);
        }

        bool isSimple() const override { return false; }    // == ! requireComposition
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, true, false>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double) const override {
            ensureCompositionIsNotEmpty(composition);
            ensureDopantIsNo(doping_amount_type);
            return make_shared<MaterialType>(Material::completeComposition(composition));
        }

        bool isSimple() const override { return false; }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, false, true>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const override {
            ensureCompositionIsEmpty(composition);
            return make_shared<MaterialType>(doping_amount_type, doping_amount);
        }

        bool isSimple() const override { return true; }
    };

    template <typename MaterialType>
    struct DelegateMaterialConstructor<MaterialType, false, false>: public MaterialConstructor {

        DelegateMaterialConstructor(const std::string& material_name): MaterialConstructor(material_name) {}

        virtual shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double) const override {
            ensureCompositionIsEmpty(composition);
            ensureDopantIsNo(doping_amount_type);
            return make_shared<MaterialType>();
        }

        bool isSimple() const override { return true; }
    };

    /**
     * Material constructor that holds other constructor or complete material object based on the provided name.
     */
    class PLASK_API ProxyMaterialConstructor: public MaterialConstructor {
        shared_ptr<Material> material;
        shared_ptr<const MaterialsDB::MaterialConstructor> constructor;
        Material::Composition composition;

      public:
        ProxyMaterialConstructor();

        ProxyMaterialConstructor(const std::string& name, const MaterialsDB& db=MaterialsDB::getDefault());

        ProxyMaterialConstructor(const shared_ptr<Material>& material);

        virtual shared_ptr<Material> operator()(const Material::Composition& comp, Material::DopingAmountType dopt, double dop) const override {
            if (material) {
                ensureCompositionIsEmpty(comp);
                ensureDopantIsNo(dopt);
                return material;
            } else if (composition.empty()) {
                return (*constructor)(comp, dopt, dop);
            } else {
                ensureCompositionIsEmpty(comp);
                return (*constructor)(composition, dopt, dop);
            }
        }

        bool isSimple() const override {
            if (material || !composition.empty() || !constructor) return true;
            return constructor->isSimple();
        }
    };

    /**
     * Create material object.
     * @param composition complete objects composition
     * @param dopant_name name of dopant (if any)
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const Material::Composition& composition, const std::string& dopant_name = "", Material::DopingAmountType doping_amount_type = Material::NO_DOPING, double doping_amount = 0.0) const;

    /**
     * Create material object.
     * @param parsed_name_with_dopant material name with dopant name in format material_name[:dopant_name], for example: "AlGaN" or "AlGaN:Mg"
     * @param composition amounts of objects, with NaN for each object for which composition was not given
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     */
    shared_ptr<Material> get(const std::string& parsed_name_with_dopant, const std::vector<double>& composition, Material::DopingAmountType doping_amount_type = Material::NO_DOPING, double doping_amount = 0.0) const;

    /**
     * Create material object.
     * @param name_with_dopant material name with dopant name in format material[:dopant_name], for example: "Al(0.2)GaN" or "Al(0.2)GaN:Mg"
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     */
    shared_ptr<Material> get(const std::string& name_with_dopant, Material::DopingAmountType doping_amount_type, double doping_amount) const;

    /**
     * Create material object.
     * @param name_with_components objects composition in format object1(amount1)...objectN(amountN), where some amounts are optional for example: "Al(0.7)GaN"
     * @param doping_descr empty string if there is no doping or description of dopant in format objectname=amount or objectname p/n=amount, for example: "Mg=7e18" or "Mg p=7e18"
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @p parsed_name_with_donor
     * @throw MaterialParseException if can't parse @p name_with_components or @p doping_descr
     */
    shared_ptr<Material> get(const std::string& name_with_components, const std::string& doping_descr) const;

    /**
     * Get constructor of material.
     * @param name_without_composition material name, without encoded parameters, in format composition[:dopant]
     * @return constructor of material or nullptr if there is no such material in database
     * @throw NoSuchMaterial if database doesn't know material with name @p name_without_composition
     */
    shared_ptr<const MaterialConstructor> getConstructor(const std::string& name_without_composition) const;

    /**
     * Create material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see get(const std::string&, const std::string&)
     * @return material with @p full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @p full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const;

    /**
     * Construct mixed material factory, for materials without dopant.
     * @param material1composition incomplate composition of first material
     * @param material2composition incomplate composition of second material, must be defined for the same objects as @p material1composition
     * @return constructed factory
     */
    shared_ptr<MixedCompositionFactory> getFactory(const Material::Composition& material1composition, const Material::Composition& material2composition) const;

    /**
     * Construct mixed material factory.
     * @param material1composition incomplate composition of first material
     * @param material2composition incomplate composition of second material, must be defined for the same objects as @p material1composition
     * @param dopant_name name of dopant, empty if there is no dopant
     * @param dopAmountType type of doping amounts, common for @p m1DopAmount and @p m2DopAmount
     * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
     * @return constructed factory
     */
    shared_ptr<MixedCompositionFactory> getFactory(const Material::Composition& material1composition, const Material::Composition& material2composition, const std::string& dopant_name,
                                        Material::DopingAmountType dopAmountType, double m1DopAmount, double m2DopAmount) const;

    /**
     * Construct mixed material factory.
     * @param material1_name_with_components composition of first material
     * @param material2_name_with_components composition of second material, must be defined for the same objects as @p material1composition
     * @param dopant_name name of dopant, common for both materials, empty if there is no dopant
     * @param dopAmountType type of doping amounts, common for @p m1DopAmount and @p m2DopAmount
     * @param m1DopAmount, m2DopAmount amounts of doping for first and second material
     * @return constructed factory created using new operator, should by delete by caller
     */
    shared_ptr<MixedCompositionFactory> getFactory(const std::string& material1_name_with_components, const std::string& material2_name_with_components,
                                        const std::string& dopant_name, Material::DopingAmountType dopAmountType, double m1DopAmount, double m2DopAmount) const;

    /**
     * Construct mixed material factory.
     * @param material1_fullname, material2_fullname materials name, with encoded parameters in format composition[:dopant], see get(const std::string& name_with_components, const std::string& dopant_descr),
     *      both must refer to the same material with the same dopant and in case of doping materials, amounts of dopants must be given in the same format
     * @return constructed factory
     */
    shared_ptr<MixedCompositionFactory> getFactory(const std::string& material1_fullname, const std::string& material2_fullname) const;

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
     * Add complex material (which require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance; must be created by operator new and material DB will call delete for it
     */
    //void addComplex(const MaterialConstructor* constructor);

    /**
     * Add complex material (which require composition parsing) to DB. Replace existing material if there is one already in DB.
     * @param constructor object which can create material instance
     */
    void addComplex(shared_ptr<MaterialConstructor> constructor);

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
            addComplex(make_shared< DelegateMaterialConstructor<MaterialType, requireComposition, requireDopant> >(name));
        else
            addSimple(make_shared< DelegateMaterialConstructor<MaterialType, requireComposition, requireDopant> >(name));
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
     * Remove simple material (which not require composition parsing) from DB.
     * @param name material name, in format name[:dopant]
     */
    void removeSimple(const std::string& name);

    /**
     * Remove complex material (which require composition parsing) from DB.
     * @param name material name, in format name[:dopant]
     */
    void removeComplex(const std::string& name);

    /**
     * Remove material from DB.
     * @param name material name, in format name[:dopant]
     * @tparam @c true only if material is complex, @c false if it's simple
     */
    template <bool isComplex>
    void remove(const std::string& name) {
        if (isComplex) removeComplex(name); else removeSimple(name);
    }

    /**
     * Remove material from DB.
     *
     * Deduce from constructors if material is either complex or simple.
     * @param name material name (with dopant after ':')
     */
    template <typename MaterialType>
    void remove(const std::string& name) { remove<Material::is_with_composition<MaterialType>::value>(name); }

    /**
     * Remove material from DB.
     *
     * Deduce from constructors if material is either complex or simple.
     * Material name is read from static field MaterialType::static_name.
     */
    template <typename MaterialType>
    void remove() { remove<Material>(MaterialType::static_name); }

    /**
     * Check if dopant is included in @p material_name.
     * @param material_name full material name
     * @return @c true only if dopant is included in @p material_name.
     */
    static bool isNameWithDopant(const std::string& material_name);

    /**
     * Check if @p material_name is name of simple material.
     * @param material_name full material name or name without dopant (only part before ':')
     * @return @c true only if @p material_name is name of simple material (does not have composition).
     */
    static bool isSimpleMaterialName(const std::string& material_name);

    /**
     * Check if a material (given without patameters) is simple.
     * @param name_without_composition material name, without encoded parameters, in format composition[:dopant]
     * @return @c true only if the material is simple
     * @throw NoSuchMaterial if database doesn't know the material with name @p name_without_composition
     */
    bool isSimple(const std::string& material_name) const;

    /**
     * Get complex material constructor object.
     * @param composition objects composition
     * @param dopant_name name of dopant (if any)
     */
    shared_ptr<const MaterialConstructor> getConstructor(const Material::Composition& composition, const std::string& dopant_name = "") const;

private:

    /**
     * Get material constructor object.
     * @param dbKey key in database
     * @param composition objects composition, empty composition for simple materials, used for error checking and mesages
     * @param dopant_name name of dopant (if any), use for error mesages
     */
    shared_ptr<const MaterialConstructor> getConstructor(const std::string& dbKey, const Material::Composition& composition, const std::string& dopant_name = "") const;

    /**
     * Create material object.
     * @param dbKey key in database
     * @param composition objects composition, empty composition for simple materials
     * @param dopant_name name of dopant (if any)
     * @param doping_amount_type type of amount of dopant, needed to interpretation of @p dopant_amount
     * @param doping_amount amount of dopant, is ignored if @p doping_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if there is no material with key @p dbKey in database
     * @see @ref Material::completeComposition
     */
    shared_ptr<Material> get(const std::string& dbKey, const Material::Composition& composition,
                             const std::string& dopant_name = "", Material::DopingAmountType doping_amount_type = Material::NO_DOPING,
                             double doping_amount = 0.0) const;

};

/*
 * Type of material source, can return material with given name.
 */
//typedef std::function<shared_ptr<Material>(const std::string& material_full_name)> MaterialsSource;
//TODO ? maybe MaterialsSource should be a class, a base class for MaterialsDB, and maybe it should has interface to append new materials

/**
 * Type of material source, can return material with given name.
 */
struct PLASK_API MaterialsSource {

    virtual ~MaterialsSource() {}

    virtual const MaterialsDB* getDB() const = 0;

    /**
     * Get material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see get(const std::string&, const std::string&)
     * @return material with @p full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @p full_name
     */
    virtual shared_ptr<Material> get(const std::string& full_name) const = 0;

    /**
     * Construct mixed material factory.
     * @param material1_fullname, material2_fullname materials name, with encoded parameters in format composition[:dopant], see get(const std::string& name_with_components, const std::string& dopant_descr),
     *      both must refer to the same material with the same dopant and in case of doping materials, amounts of dopants must be given in the same format
     * @return constructed factory created using operator new, should by delete by caller
     */
    virtual shared_ptr<MaterialsDB::MixedCompositionFactory> getFactory(const std::string& material1_fullname, const std::string& material2_fullname) const = 0;
};

struct PLASK_API MaterialsSourceDB: public MaterialsSource {

    const MaterialsDB& db;

    MaterialsSourceDB(const MaterialsDB& db): db(db) {}

    virtual const MaterialsDB* getDB() const { return &db; }

    /**
     * Get material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see get(const std::string&, const std::string&)
     * @return material with @p full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @p full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const { return db.get(full_name); }

    /**
     * Construct mixed material factory.
     * @param material1_fullname, material2_fullname materials name, with encoded parameters in format composition[:dopant], see get(const std::string& name_with_components, const std::string& dopant_descr),
     *      both must refer to the same material with the same dopant and in case of doping materials, amounts of dopants must be given in the same format
     * @return constructed factory created using operator new, should by delete by caller
     */
    virtual shared_ptr<MaterialsDB::MixedCompositionFactory> getFactory(const std::string& material1_fullname, const std::string& material2_fullname) const {
        return db.getFactory(material1_fullname, material2_fullname);
    }
};

}   // namespace plask

#endif // PLASK__MATERIAL_DB_H
