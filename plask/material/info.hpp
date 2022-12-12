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
#ifndef PLASK__MATERIAL_INFO_H
#define PLASK__MATERIAL_INFO_H

/** @file
This file contains classes which stores meta-informations about materials.
*/

#include <string>
#include <map>
#include <vector>
#include <utility>

#include <boost/tokenizer.hpp>

#include "../optional.hpp"

#include <plask/config.hpp>   // for PLASK_API

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
        taue,       ///< monomolecular electrons lifetime [ns]
        tauh,       ///< monomolecular holes lifetime [ns]
        Ce,         ///< Auger recombination coefficient for electrons
        Ch,         ///< Auger recombination coefficient for holes
        e13,        ///< piezoelectric constant
        e15,        ///< piezoelectric constant
        e33,        ///< piezoelectric constant
        c13,        ///< elastic constant
        c33,        ///< elastic constant
        Psp,        ///< spontaneous polarization
        y1,         ///< Luttinger parameter 1
        y2,         ///< Luttinger parameter 2
        y3          ///< Luttinger parameter 3
    };

    /// Names of the properties
    static const char* PROPERTY_NAME_STRING[55];

    static PROPERTY_NAME parsePropertyName(const std::string& name);

    /// Names of arguments for which we need to give the ranges
    enum ARGUMENT_NAME {
        T,          ///< temperature
        e,          ///< strain
        lam,        ///< wavelength
        n,          ///< carriers concentration
        h,          ///< thickness
        doping,     ///< doping
        point       ///< point
    };

    /// Names of the arguments
    static const char* ARGUMENT_NAME_STRING[7];

    static ARGUMENT_NAME parseArgumentName(const std::string& name);

    /**
     * Represent link ("see also") to property in class.
     */
    struct PLASK_API Link {
        ///Class name.
        std::string className;
        ///Property.
        PROPERTY_NAME property;
        ///Link note.
        std::string note;

        Link(std::string className, PROPERTY_NAME property, std::string note = std::string())
            : className(className), property(property), note(note) {}

        /**
         * Construct a link from a string.
         * @param to_parse the string to parse, in form "class_name.property info"
         */
        Link(const std::string& to_parse);

        /**
         * Construct a string in format "class_name.property info".
         * @return the string constructed
         */
        std::string str() const;
    };

    /// Collect information about material property.
    class PLASK_API PropertyInfo {

        /// Other infos about property
        std::string _info;

        /**
         * Append info to string returned by get().
         *
         * If @p get() is not empty it appends end-line between @p get() and @p info.
         * @param infoToAdd info to append
         * @return *this
         */
        PropertyInfo& add(const std::string& infoToAdd) {
            if (_info.empty()) _info = infoToAdd; else (_info += '\n') += infoToAdd;
            return *this;
        }

    public:

        boost::tokenizer<boost::char_separator<char>> eachLine() const;

        std::vector<std::string> eachOfType(const std::string& type) const;

        typedef std::pair<double, double> ArgumentRange;

        /// Construct info with given info.
        PropertyInfo(std::string info = ""): _info(std::move(info)) {}

        /// Returned by getArgumentRange if there is no range for given argument, holds two NaNs
        static const ArgumentRange NO_RANGE;

        /**
         * Set info on this material property.
         * @param new_info info
         * @return *this
         */
        PropertyInfo& set(const std::string& new_info) { this->_info = new_info; return *this; }

        /**
         * Get info on this material property.
         * @return info on this material property
         */
        const std::string& get() const { return _info; }

        /**
         * Get bibliography source of this material property. One source per line.
         * @return bibliography source of this material property
         */
        std::string getSource() const;

        /**
         * Get note for this material property. One source per line.
         * @return note this material property
         */
        std::string getNote() const;

        /**
         * Get the range of argument's values for which the calculation method is known to works fine.
         * @param argument name of requested argument
         * @return range (NO_RANGE if the information is not available)
         */
        ArgumentRange getArgumentRange(ARGUMENT_NAME argument) const;

        /**
         * Get array of "see also" links.
         *
         * It finds them in info. It tries to parse ech line which starts with "see:".
         * @return array of "see also" links
         */
        std::vector<Link> getLinks() const;

        /**
         * Append bibliography source to the info (and the string returned by getSource()).
         * @param sourceToAdd bibliography source to append
         * @return *this
         */
        PropertyInfo& addSource(const std::string& sourceToAdd) {
            return add("source: " + sourceToAdd);
        }

        /**
         * Append note to the info (and the string returned by getSource()).
         * @param sourceToAdd bibliography source to append
         * @return *this
         */
        PropertyInfo& addNote(const std::string& sourceToAdd) {
            return add("note: " + sourceToAdd);
        }

        /**
         * Set the range of argument's values for which the calculation method is known to works fine.
         * @param argument name of argument
         * @param range range of values for which calculation method is known to works fine (NO_RANGE if the information is not available)
         * @return *this
         */
        PropertyInfo& setArgumentRange(ARGUMENT_NAME argument, ArgumentRange range);

        /**
         * Set the range of argument's values for which the calculation method is known to works fine.
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
        PropertyInfo& addLink(const Link& link) { return add("see: " + link.str()); }
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
    plask::optional<PropertyInfo> getPropertyInfo(PROPERTY_NAME property) const;

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

      public:

        /**
         * Get default database of materials' meta-informations.
         * @return default database of materials' meta-informations
         */
        static DB& getDefault();

        /**
         * Clear the database
         */
        void clear() {
            materialInfo.clear();
        }

        /**
         * Update with values from different database
         * \param src source database
        */
        void update(const DB& src) {
            for (const auto& item: src.materialInfo) {
                materialInfo[item.first] = item.second;
            }
        }

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
        plask::optional<MaterialInfo> get(const std::string& materialName, bool with_inherited_info = true) const;

        /**
         * Get meta-informations about material's property from database.
         * @param materialName, propertyName name of material and its property to get information about
         * @param with_inherited_info if true (default) returned object will consists also with information inherited from parent, grand-parent, etc. materials
         * @return meta-informations about material's property from database, no value if meta-informations of requested material are not included in data-base
         */
        plask::optional<MaterialInfo::PropertyInfo> get(const std::string& materialName, PROPERTY_NAME propertyName, bool with_inherited_info = true) const;

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

        RegisterProperty& info(const std::string& info) { property.add(info); return *this; }

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

struct MINote {
    std::string value;
    MINote(const std::string& value): value(value) {}
    void set(MaterialInfo::PropertyInfo& p) const { p.addNote(value); }
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
