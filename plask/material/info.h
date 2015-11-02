#ifndef PLASK__MATERIAL_INFO_H
#define PLASK__MATERIAL_INFO_H

/** @file
This file contains classes which stores meta-informations about materials.
*/

#include <string>
#include <map>
#include <vector>
#include <utility>

#include <boost/optional.hpp>

#include <plask/config.h>   // for PLASK_API

namespace plask {

/**
 * Collect meta-informations about material.
 *
 * It also namespace for all class connected with collecting meta-informations about materials, like material informations database, etc.
 */
struct PLASK_API MaterialInfo {

    enum PROPERTY_NAME {
        kind,       ///< material kind
        lattC,      ///< lattice constant
        Eg,         ///< energy gap
        CB,         ///< conduction band offset
        VB,         ///< valence band offset
        Dso,        ///< split-off energy
        Mso,        ///< split-off mass
        Me,         ///< electron effective mass
        Mhh,        ///< heavy-hole effective mass
        Mlh,        ///< light-hole effective mass
        Mh,         ///< hole effective mass
        ac,         ///< hydrostatic deformation potential for the conduction band
        av,         ///< hydrostatic deformation potential for the valence band
        b,          ///< shear deformation potential
        d,          ///< shear deformation potential
        c11,        ///< elastic constant
        c12,        ///< elastic constant
        c44,        ///< elastic constant
        eps,        ///< dielectric constant
        chi,        ///< electron affinity
        Na,         ///< acceptor concentration
        Nd,         ///< donor concentration
        Ni,         ///< intrinsic carrier concentration
        Nf,         ///< free carrier concentration
        EactD,      ///< donor ionisation energy
        EactA,      ///< acceptor ionisation energy
        mob,        ///< mobility
        cond,       ///< electrical conductivity
        condtype,   ///< conductivity type
        A,          ///< monomolecular recombination coefficient
        B,          ///< radiative recombination coefficient
        C,          ///< Auger recombination coefficient
        D,          ///< ambipolar diffusion coefficient
        thermk,     ///< thermal conductivity
        dens,       ///< density
        cp,         ///< specific heat at constant pressure
        nr,         ///< refractive index
        absp,       ///< absorption coefficient alpha
        Nr,         ///< complex refractive index
        NR,         ///< complex refractive index tensor

        mobe,       ///< electron mobility
        mobh,       ///< hole mobility
        Ae,         ///< monomolecular recombination coefficient for electrons
        Ah,         ///< monomolecular recombination coefficient for holes
        Ce,         ///< Auger recombination coefficient for electrons
        Ch,         ///< Auger recombination coefficient for holes
        e13,        ///< piezoelectric constant
        e15,        ///< piezoelectric constant
        e33,        ///< piezoelectric constant
        c13,        ///< elastic constant
        c33,        ///< elastic constant
        Psp         ///< spontaneous polarization
    };

    /// Names of the properties
    static const char* PROPERTY_NAME_STRING[];

    /// Names of arguments for which we need to give the ranges
    enum ARGUMENT_NAME {
        T,          ///< temperature
        e,          ///< strain
        wl,         ///< wavelength
        n,          ///< carriers concentration
        h,          ///< thickness
        doping      ///< doping
    };

    /// Names of the arguments
    static const char* ARGUMENT_NAME_STRING[];

    /**
     * Represent link ("see also") to property in class.
     */
    struct PLASK_API Link {
        ///Class name.
        std::string className;
        ///Property.
        PROPERTY_NAME property;
        ///Link comment.
        std::string comment;

        Link(std::string className, PROPERTY_NAME property, std::string comment = std::string())
            : className(className), property(property), comment(comment) {}
    };

    ///Collect information about material property.
    struct PLASK_API PropertyInfo {

        typedef std::pair<double, double> ArgumentRange;

    private:

        /// Property arguments constraints (ranges)
        std::map<ARGUMENT_NAME, ArgumentRange> _argumentRange;

        /// See also links to properties in another class.
        std::vector<Link> _links;

        /// Information about source of property calclulation algorithm
        std::string _source;

        ///Other comments about property
        std::string _comment;

        /**
         * Append @p what to @p where.
         * If @p where is not empty it append end-line between @p where and @p what.
         * @param where string to change
         * @param what string to append to @p where
         */
        static void addToString(std::string& where, const std::string& what) {
            if (where.empty()) where = what; else (where += '\n') += what;
        }

    public:

        /// Returned by getArgumentRange if there is no range for given argument, holds two NaNs
        static const ArgumentRange NO_RANGE;

        /**
         * Set bibliography source of calculation method of this material property.
         * @param new_source bibliography source
         * @return *this
         */
        PropertyInfo& setSource(const std::string& new_source) { this->_source = new_source; return *this; }

        /**
         * Get bibliography source of calculation method of this material property.
         * @return bibliography source of calculation method of this material property
         */
        const std::string& getSource() const { return _source; }

        /**
         * Set comment for this material property.
         * @param new_comment comment
         * @return *this
         */
        PropertyInfo& setComment(const std::string& new_comment) { this->_comment = new_comment; return *this; }

        /**
         * Get comment for this material property.
         * @return comment for this material property
         */
        const std::string& getComment() const { return _comment; }

        /**
         * Get the range of argument's values for which calculation method is known to works fine.
         * @param argument name of requested argument
         * @return range (NO_RANGE if the information is not available)
         */
        const ArgumentRange& getArgumentRange(ARGUMENT_NAME argument) const;

        /**
         * Get array of "see also" links.
         * @return array of "see also" links
         */
        const std::vector<Link>& getLinks() const { return _links; }

        /**
         * Append bibliography source to string returned by getSource().
         * @param sourceToAdd bibliography source to append
         * @return *this
         */
        PropertyInfo& addSource(const std::string& sourceToAdd) { addToString(this->_source, sourceToAdd); return *this; }

        /**
         * Append comment to string returned by getComment().
         * @param commentToAdd comment to append
         * @return *this
         */
        PropertyInfo& addComment(const std::string& commentToAdd) { addToString(this->_comment, commentToAdd); return *this; }

        /**
         * Set the range of argument's values for which calculation method is known to works fine.
         * @param argument name of argument
         * @param range range of values for which calculation method is known to works fine (NO_RANGE if the information is not available)
         * @return *this
         */
        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, ArgumentRange range);

        /**
         * Set the range of argument's values for which calculation method is known to works fine.
         * @param argument name of argument
         * @param from, to range of values for which calculation method is known to works fine
         * @return *this
         */
        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, double from, double to) {
            return setArgumentRange(argument, ArgumentRange(from, to));
        }

        /**
         * Add link to array of "see also" links.
         * @param link link to add
         * @return *this
         */
        PropertyInfo& addLink(const Link& link) { _links.push_back(link); return *this; }

        /**
         * Move link to array of "see also" links.
         * @param link link to add (move)
         * @return *this
         */
        PropertyInfo& addLink(Link&& link) { _links.push_back(std::move(link)); return *this; }
    };

    /// Name of parent class
    std::string parent;

    private:
    /// Information about properties.
    std::map<PROPERTY_NAME, PropertyInfo> propertyInfo;

    public:

    /**
     * Override all information about this by informations from @p to_override.
     *
     * It can be used to override the information in this by its subclass informations.
     * @param to_override
     */
    void override(const MaterialInfo &to_override);

    /**
     * Get property info object (add new, empty one if there is no information about property).
     * @param property
     */
    PropertyInfo& operator()(PROPERTY_NAME property);

    /**
     * Get info about the @p property of this.
     * @param property
     * @return info about the @p property of this
     */
    boost::optional<PropertyInfo> getPropertyInfo(PROPERTY_NAME property) const;

    /// Iterator over properties
    typedef std::map<PROPERTY_NAME, PropertyInfo>::iterator iterator;

    /// Const iterator over properties
    typedef std::map<PROPERTY_NAME, PropertyInfo>::const_iterator const_iterator;

    /**
     * Get begin iterator over properties.
     * @return begin iterator over properties
     */
    iterator begin() { return propertyInfo.begin(); }

    /**
     * Get const begin iterator over properties.
     * @return const begin iterator over properties
     */
    const_iterator begin() const { return propertyInfo.begin(); }

    /**
     * Get end iterator over properties.
     * @return end iterator over properties
     */
    iterator end() { return propertyInfo.end(); }

    /**
     * Get const end iterator over properties.
     * @return const end iterator over properties
     */
    const_iterator end() const { return propertyInfo.end(); }

    //const PropertyInfo& operator()(PROPERTY_NAME property) const;

    /// Material info database
    class PLASK_API DB {

        /// Material name -> material information
        std::map<std::string, MaterialInfo> materialInfo;

        /// Parent database which is requested for all materials not described in this database
        const DB* parent;

    public:

        /**
         * Create database.
         * @param parent parent database, which is requested for all materials not described in this database.
         */
        explicit DB(const DB* parent = nullptr): parent(parent) {}

        /**
         * Get default database of materials' meta-informations.
         * @return default database of materials' meta-informations
         */
        static DB& getDefault();

        /**
         * Add meta-informations about material to database.
         * @param materialName name of material to add
         * @param parentMaterial parent material, from which all properties all inherited (some may be overwritten)
         * @return material info object which allow to fill detailed information
         */
        MaterialInfo& add(const std::string& materialName, const std::string& parentMaterial);

        /**
         * Add meta-informations about material to database.
         * @param materialName name of material to add
         * @return material info object which allow to fill detailed information
         */
        MaterialInfo& add(const std::string& materialName);

        /**
         * Get meta-informations about material from database.
         * @param materialName name of material to get information about
         * @param with_inherited_info if true (default) returned object will consists also with information inherited from parent, grand-parent, etc. materials
         * @return meta-informations about material with name @p materialName, no value if meta-informations of requested material are not included in data-base
         */
        boost::optional<MaterialInfo> get(const std::string& materialName, bool with_inherited_info = true) const;

        /**
         * Get meta-informations about material's property from database.
         * @param materialName, propertyName name of material and its property to get information about
         * @param with_inherited_info if true (default) returned object will consists also with information inherited from parent, grand-parent, etc. materials
         * @return meta-informations about material's property from database, no value if meta-informations of requested material are not included in data-base
         */
        boost::optional<MaterialInfo::PropertyInfo> get(const std::string& materialName, PROPERTY_NAME propertyName, bool with_inherited_info = true) const;

        /// iterator over materials' meta-informations
        typedef std::map<std::string, MaterialInfo>::iterator iterator;

        /// const iterator over materials' meta-informations
        typedef std::map<std::string, MaterialInfo>::const_iterator const_iterator;

        /**
         * Get begin iterator over materials' meta-informations.
         * @return begin iterator over  materials' meta-information
         */
        iterator begin() { return materialInfo.begin(); }

        /**
         * Get const begin iterator over materials' meta-informations.
         * @return const begin iterator over  materials' meta-information
         */
        const_iterator begin() const { return materialInfo.begin(); }

        /**
         * Get end iterator over materials' meta-informations.
         * @return end iterator over  materials' meta-information
         */
        iterator end() { return materialInfo.end(); }

        /**
         * Get const end iterator over materials' meta-informations.
         * @return const end iterator over  materials' meta-information
         */
        const_iterator end() const { return materialInfo.end(); }

    };

    /**
     * Helper which allow to add (do this in constructor) information about material to default material meta-info database.
     */
    class PLASK_API Register {

        void set(PropertyInfo&) {}

        template <typename Setter1, typename ...Setters>
        void set(PropertyInfo& i, const Setter1& setter1, const Setters&... setters) {
            setter1.set(i);
            set(i, setters...);
        }

    public:
        Register(const std::string& materialName, const std::string& parentMaterial) {
            MaterialInfo::DB::getDefault().add(materialName, parentMaterial);
        }

        template <typename ...PropertySetters>
        Register(const std::string& materialName, const std::string& parentMaterial, PROPERTY_NAME property, const PropertySetters&... propertySetters) {
            set(MaterialInfo::DB::getDefault().add(materialName, parentMaterial)(property), propertySetters...);
        }

        Register(const std::string& materialName) {
            MaterialInfo::DB::getDefault().add(materialName);
        }

        template <typename ...PropertySetters>
        Register(const std::string& materialName, PROPERTY_NAME property, const PropertySetters&... propertySetters) {
            set(MaterialInfo::DB::getDefault().add(materialName)(property), propertySetters...);
        }

        /*template <typename MaterialType, typename ParentType, typename ...PropertySetters>
        Register(const PropertySetters&... propertySetters) {
            set(MaterialInfo::DB::getDefault().add(MaterialType::NAME, ParentType::NAME), propertySetters...);
        }

        template <typename MaterialType, typename ...PropertySetters>
        Register(const PropertySetters&... propertySetters) {
            set(MaterialInfo::DB::getDefault().add(MaterialType::NAME), propertySetters...);
        }*/

    };

    /*struct RegisterProperty {

        PropertyInfo& property;

        RegisterProperty(Register& material, PROPERTY_NAME property)
            : property(material(property)) {}

        RegisterProperty(std::string& material_name, PROPERTY_NAME property)
            : property(MaterialInfo::DB::getDefault().add(material_name)(property)) {}

        template <typename MaterialType>
        RegisterProperty(PROPERTY_NAME property)
            : property(MaterialInfo::DB::getDefault().add(MaterialType::NAME)(property)) {}

        RegisterProperty& source(const std::string& source) { property.addSource(source); return *this; }

        RegisterProperty& comment(const std::string& comment) { property.addComment(comment); return *this; }

        RegisterProperty& argumentRange(ARGUMENT_NAME argument, PropertyInfo::ArgumentRange range) {
            property.setArgumentRange(argument, range); return *this;
        }

        RegisterProperty& argumentRange(ARGUMENT_NAME argument, double from, double to) {
            property.setArgumentRange(argument, from, to); return *this;
        }

    };*/

};

struct MISource {
    std::string value;
    MISource(const std::string& value): value(value) {}
    void set(MaterialInfo::PropertyInfo& p) const { p.addSource(value); }
};

struct MIComment {
    std::string value;
    MIComment(const std::string& value): value(value) {}
    void set(MaterialInfo::PropertyInfo& p) const { p.addComment(value); }
};

struct MIArgumentRange {
    MaterialInfo::ARGUMENT_NAME arg; double from, to;
    MIArgumentRange(MaterialInfo::ARGUMENT_NAME arg, double from, double to): arg(arg), from(from), to(to) {}
    void set(MaterialInfo::PropertyInfo& p) const { p.setArgumentRange(arg, from, to); }
};

struct MISee {
    MaterialInfo::Link value;
    template<typename ...Args> MISee(Args&&... params): value(std::forward<Args>(params)...) {}
    void set(MaterialInfo::PropertyInfo& p) const { p.addLink(value); }
};

template <typename materialClass>
struct MISeeClass {
    MaterialInfo::Link value;
    template<typename ...Args> MISeeClass(Args&&... params): value(materialClass::NAME, std::forward<Args>(params)...) {}
    void set(MaterialInfo::PropertyInfo& p) const { p.addLink(value); }
};

}

#define MI_PARENT(material, parent) static plask::MaterialInfo::Register __materialinfo__parent__  ## material(material::NAME, parent::NAME);
#define MI_PROPERTY(material, property, ...) static plask::MaterialInfo::Register __materialinfo__property__ ## material ## property(material::NAME, plask::MaterialInfo::property, ##__VA_ARGS__);
#define MI_PARENT_PROPERTY(material, parent, property, ...) static plask::MaterialInfo::Register __materialinfo__parent__property__ ## material ## property(material::NAME, parent::NAME, plask::MaterialInfo::property, ##__VA_ARGS__);

#endif // PLASK__MATERIAL_INFO_H
