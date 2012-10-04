#ifndef PLASK__MATERIAL_INFO_H
#define PLASK__MATERIAL_INFO_H

/** @file
This file includes classes which stores meta-informations about materials.
*/

#include <string>
#include <map>
#include <vector>
#include <utility>

namespace plask {

/**
 * Collect meta-informations about material.
 *
 * It also namespace for all class connected with collecting meta-informations about materials, like material informations database, etc.
 */
struct MaterialInfo {

    enum PROPERTY_NAME {
        kind,       ///< material kind
        lattC,      ///< lattice constant
        Eg,         ///< energy gap
        CBO,        ///< conduction band offset
        VBO,        ///< valence band offset
        Dso,        ///< split-off energy
        Mso,        ///< split-off mass
        Me,         ///< electron effective mass
        Mhh,        ///< heavy-hole effective mass
        Mlh,        ///< light-hole effective mass
        Mh,         ///< hole effective mass
        eps,        ///< dielectric constant
        chi,        ///< electron affinity
        Nc,         ///< effective density of states in the conduction band
        Nv,         ///< effective density of states in the valence band
        Ni,         ///< intrinsic carrier concentration
        Nf,         ///< free carrier concentration
        EactD,      ///< donor ionisation energy
        EactA,      ///< acceptor ionisation energy
        mob,        ///< mobility
        cond,       ///< electrical conductivity
        condType,   ///< conductivity type
        A,          ///< monomolecular recombination coefficient
        B,          ///< radiative recombination coefficient
        C,          ///< Auger recombination coefficient
        D,          ///< ambipolar diffusion coefficient
        thermCond,  ///< thermal conductivity
        dens,       ///< density
        specHeat,   ///< specific heat at constant pressure
        nr,         ///< refractive index
        absp,       ///< absorption coefficient alpha
        Nr          ///< refractive index
    };

    ///Names of arguments for which range we need give
    enum ARGUMENT_NAME {
        T,          ///<temperature
        thickness,  ///<thickness
        wl,         ///<wavelength
        doping      ///<doping
    };

    /**
     * Represent link ("see also") to property in class.
     */
    struct Link {
        ///Class name.
        std::string className;
        ///Property.
        PROPERTY_NAME property;
        ///Link comment.
        std::string comment;

        Link(std::string className, PROPERTY_NAME property, std::string comment = std::string())
            : className(className), property(property), comment(comment) {}
    };

    ///Name of parent class
    std::string parent;

    ///Collect information about material property.
    struct PropertyInfo {

        typedef std::pair<double, double> ArgumentRange;

    private:

        ///Property arguments constraints (ranges)
        std::map<ARGUMENT_NAME, ArgumentRange> _argumentRange;

        ///See also links to properties in another class.
        std::vector<Link> _links;

        ///Information about source of property calclulation algorithm
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

        /// returned by getArgumentRange if there is no range for given argument, hold two NaNs
        static const ArgumentRange NO_RANGE;

        PropertyInfo& setSource(const std::string& new_source) { this->_source = new_source; return *this; }

        const std::string& getSource() const { return _source; }

        PropertyInfo& setComment(const std::string& new_comment) { this->_comment = new_comment; return *this; }

        const std::string& getComment() const { return _comment; }

        const ArgumentRange& getArgumentRange(ARGUMENT_NAME argument);

        const std::vector<Link>& getLinks() const { return _links; }

        PropertyInfo& addSource(const std::string& sourceToAdd) { addToString(this->_source, sourceToAdd); return *this; }

        PropertyInfo& addComment(const std::string& commentToAdd) { addToString(this->_comment, commentToAdd); return *this; }

        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, ArgumentRange range);

        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, double from, double to) {
            return setArgumentRange(argument, ArgumentRange(from, to));
        }

        PropertyInfo& addLink(const Link& link) { _links.push_back(link); return *this; }

        PropertyInfo& addLink(Link&& link) { _links.push_back(link); return *this; }
    };

    std::map<PROPERTY_NAME, PropertyInfo> propertyInfo;

    /**
     * Get property info object (add new, empty one if there is no information about property).
     */
    PropertyInfo& operator()(PROPERTY_NAME property);

    //const PropertyInfo& operator()(PROPERTY_NAME property) const;

    ///Material info database
    class DB {

    public:

        static DB& getDefault();

        ///Material name -> material information
        std::map<std::string, MaterialInfo> materialInfo;

        /**
         * Add meta-informations about material to database.
         * @param materialName name of material to add
         * @param parentMaterial parent material, from which all properties all inharited (some may be overwritten)
         * @return material info object which allow to fill detailed information
         */
        MaterialInfo& add(const std::string& materialName, const std::string& parentMaterial);

        MaterialInfo& add(const std::string& materialName);

    };

    /**
     * Helper which allow to add (do this in constructor) information about material to default material meta-info database.
     */
    class Register {

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
