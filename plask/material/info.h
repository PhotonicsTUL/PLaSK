#ifndef PLASK__MATERIAL_INFO_H
#define PLASK__MATERIAL_INFO_H

/** @file
This file includes classes which stores meta-informations about materials.
*/

#include <string>
#include <map>
#include <utility>

namespace plask {

/**
 * Collect meta-informations about material.
 *
 * It also namespace for all class connected with collecting meta-informations about materials, like material informations database, etc.
 */
struct MaterialInfo {

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

    ///Collect information about material property.
    struct PropertyInfo {

        typedef std::pair<double, double> ArgumentRange;

    private:

        ///Property arguments constraints (ranges)
        std::map<ARGUMENT_NAME, ArgumentRange> _argumentRange;

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

        ///returned by getArgumentRange if there is no range for given argument, hold two NaNs
        static const ArgumentRange NO_RANGE;

        PropertyInfo& setSource(const std::string& new_source) { this->_source = new_source; return *this; }

        const std::string& getSource() const { return _source; }

        PropertyInfo& setComment(const std::string& new_comment) { this->_comment = new_comment; return *this; }

        const std::string& getComment() const { return _comment; }

        const ArgumentRange& getArgumentRange(ARGUMENT_NAME argument);

        PropertyInfo& addSource(const std::string& sourceToAdd) { addToString(this->_source, sourceToAdd); return *this; }

        PropertyInfo& addComment(const std::string& commentToAdd) { addToString(this->_comment, commentToAdd); return *this; }

        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, ArgumentRange range);

        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, double from, double to) {
            return setArgumentRange(argument, ArgumentRange(from, to));
        }

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

        MaterialInfo& materialInfo;

    public:
        Register(const std::string& materialName, const std::string& parentMaterial)
            : materialInfo(MaterialInfo::DB::getDefault().add(materialName, parentMaterial)) {}

        Register(const std::string& materialName)
            : materialInfo(MaterialInfo::DB::getDefault().add(materialName)) {}

        template <typename MaterialType, typename ParentType>
        Register()
            : materialInfo(MaterialInfo::DB::getDefault().add(MaterialType::NAME, ParentType::NAME)) {}

        template <typename MaterialType>
        Register()
            : materialInfo(MaterialInfo::DB::getDefault().add(MaterialType::NAME)) {}

        PropertyInfo& operator()(PROPERTY_NAME property) {
            return materialInfo(property);
        }

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

}

#endif // PLASK__MATERIAL_INFO_H
