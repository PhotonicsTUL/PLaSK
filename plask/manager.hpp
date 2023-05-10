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
#include "solver.hpp"

#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file contains:
- plask::Manager class.
*/

#include <string>
#include <map>
#include <set>
#include <deque>
#include <boost/filesystem.hpp>

#include "utils/xml/reader.hpp"

#include "math.hpp"
#include "mesh/mesh.hpp"
#include "material/db.hpp"
#include "geometry/path.hpp"
#include "geometry/space.hpp"

namespace plask {

class Solver;

class GeometryReader;

/**
 * Geometry manager features:
 * - read/write geometries,
 * - allow for access to geometry objects (also by names).
 *
 * @see @ref geometry
 */
struct PLASK_API Manager {

    template <typename T>
    struct Map: std::map<std::string, T> {
        typename std::map<std::string, T>::iterator find(const std::string& key) {
            std::string name = key;
            std::replace(name.begin(), name.end(), '-', '_');
            return std::map<std::string, T>::find(name);
        }

        typename std::map<std::string, T>::const_iterator find(const std::string& key) const {
            std::string name = key;
            std::replace(name.begin(), name.end(), '-', '_');
            return std::map<std::string, T>::find(name);
        }
    };

    /// Throw exception with information that loading from external sources is not supported or disallowed.
    static void disallowExternalSources(Manager& PLASK_UNUSED(manager), const std::string& url, const std::string& section) {
        throw Exception("Can't load section \"{0}\" from \"{1}\". Loading from external sources is not supported or disallowed.", section, url); }

    /// Allow to support reading some sections from other files.
    struct PLASK_API ExternalSourcesFromFile {

        /// Current file name.
        boost::filesystem::path originalFileName;

        /// Name of section which is just read
        std::string currentSection;

        /// Names of files from which current section is read (detect circular reference).
        ExternalSourcesFromFile* prev;

        bool hasCircularRef(boost::filesystem::path& fileName, const std::string& section) {
            if (!currentSection.empty() || currentSection != section) return false;
            if (fileName == originalFileName) return true;
            return prev != 0 && prev->hasCircularRef(fileName, section);
        }

        /**
         * Create ExternalSourcesFromFile which doesn't support relative file names
         * (original file name is not known).
         */
        ExternalSourcesFromFile(): prev(nullptr) {}

        /**
         * Create ExternalSourcesFromFile which support relative file names.
         * @param originalFileName name of file from which XML is read now
         * @param currentSection name of current section
         * @param prev previous source (needed to cycle detection)
         */
        ExternalSourcesFromFile(const boost::filesystem::path& originalFileName,
                                const std::string& currentSection = std::string(),
                                ExternalSourcesFromFile* prev = nullptr)
            : originalFileName(originalFileName), currentSection(currentSection), prev(prev) {}

        void operator()(Manager& manager, const std::string& url, const std::string& section);

    };

private:

    /// Map holding information about global solvers names stored in \p solvers.lst
    std::map<std::string, std::map<std::string, std::string>> global_solver_names;

    /// @return @c true
    static bool acceptAllSections(const std::string&) { return true; }

    /*struct LoadFunCallbackT {
        std::pair< XMLReader, std::unique_ptr<LoadFunCallbackT> > get(const std::string& url) const {
            throw Exception("Can't load section from \"{0}\". Loading from external sources is not supported or disallowed.", url);
        }
    };*/

    /// Load section from external URL, throw exception in case of errors, takes as parameters: manager, URL and section name.
    typedef std::function<void(Manager&, const std::string&, const std::string&)> LoadFunCallbackT;

    /**
     * Try load section from external location.
     *
     * Check if current tag in @p reader has "from" attribute, and if it has, load section represented by this tag from external location using @p load_from.
     * @param reader XML data source
     * @param load_from
     */
    bool tryLoadFromExternal(XMLReader& reader, const LoadFunCallbackT& load_from);

  protected:

    /**
     * Load solver from file.
     * @param category, lib, solver_name solver parameters
     * @param name solver name
     * @return loaded solver
     */
    virtual shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name);


    /**
     * Load binary material library to default database
     * @param reader reader to read from
     */
    void loadMaterialLib(XMLReader& reader);

    /**
     * Load one material using @p reader to default database.
     * @param reader reader to read from
     */
    virtual void loadMaterial(XMLReader& reader);

  public:

    static constexpr const char* TAG_NAME_ROOT = "plask";           ///< name of root XML tag
    static constexpr const char* TAG_NAME_DEFINES = "defines";      ///< name of XML tag of section with const definitions
    static constexpr const char* TAG_NAME_MATERIALS = "materials";  ///< name of XML tag of section with materials
    static constexpr const char* TAG_NAME_GEOMETRY = "geometry";    ///< name of XML tag of section with geometry
    static constexpr const char* TAG_NAME_GRIDS = "grids";          ///< name of XML tag of section with grids (meshes and generators)
    static constexpr const char* TAG_NAME_SOLVERS = "solvers";      ///< name of XML tag of section with solvers
    static constexpr const char* TAG_NAME_CONNECTS = "connects";    ///< name of XML tag of section with provides-receiver connections
    static constexpr const char* TAG_NAME_SCRIPT = "script";        ///< name of XML tag of section with (python) script

    static constexpr const char* XML_AXES_ATTR = "axes";            ///< name of axes attribute in XML

    /// Allow to access path hints by name.
    Map<PathHints> pathHints;

    /// Roots objects, geometries.
    std::vector<shared_ptr<Geometry>> roots;

    /// Geometries and geometry objects by name.
    Map<shared_ptr<GeometryObject>> geometrics;

    /// Meshes by name.
    Map<shared_ptr<MeshBase>> meshes;

    /// Solvers by name.
    Map<shared_ptr<Solver>> solvers;

    /// Boundaries places by name.
    //TODO? move to special modules reader class to have more local scope?
    Map<boost::any> boundaries;

    /// Script read from file
    std::string script;

    /// Line in which script begins
    unsigned scriptline;

    /// Current or default names of axis.
    const AxisNames* axisNames;

    /// Flag indicating if unknown materials are allowed
    bool draft;

    /// Errors which occurred during loading (only in draft mode)
    std::deque<std::pair<int,std::string>> errors;

    /**
     * Save non-critical error
     * \param mesg error message
     * \param line line in XML file where error occurred
     */
    void pushError(const std::string& mesg, int line=-1) {
        errors.push_back(std::make_pair(line, mesg));
    }

    /**
     * Save non-critical error
     * \param error error
     * \param line line in XML file where error occurred
     */
    void pushError(const std::runtime_error& error, int line=-1) {
        pushError(error.what(), line);
    }

    /**
     * Save non-critical error
     * \param error error
     * \param line line in XML file where error occurred
     */
    void pushError(const XMLException& error, int line=-1) {
        if (line == -1) line = error.line;
        pushError(error.what(), line);
    }

    /**
     * Throw error if not in draft mode.
     * \param error error to throw
     * \param line line in XML file where error occurred
     */
    template <typename ErrorType>
    void throwErrorIfNotDraft(ErrorType error, int line=-1) {
        if (!draft) throw error;
        else pushError(error, line);
    }

    /**
     * Get current axis name.
     * @param axis_index axis index
     * @return name of axis which have an @p axis_index
     */
    std::string getAxisName(std::size_t axis_index) { return axisNames->operator [](axis_index); }

    /**
     * Get current lon direction axis name.
     * @return name of lon axis
     */
    std::string getAxisLongName() { return getAxisName(axis::lon_index); }

    /**
     * Get current tran direction axis name.
     * @return name of tran axis
     */
    std::string getAxisTranName() { return getAxisName(axis::tran_index); }

    /**
     * Get current up direction axis name.
     * @return name of up axis
     */
    std::string getAxisVertName() { return getAxisName(axis::up_index); }

    /**
     * Set axis name from current reader tag, set it in manager as current,
     * and restore old axisNames value when out of the scope.
     */
    class PLASK_API SetAxisNames {
        Manager& manager;
        const AxisNames* old;

      public:

        /**
         * Set axes names to @p names.
         * @param manager manager to change
         * @param names new axes names
         */
        SetAxisNames(Manager& manager, const AxisNames* names);

        /**
         * Set axes names to one read from tag attribute.
         *
         * Does not change the axes names in case of lack the attribute.
         * @param manager manager to change
         * @param source XML data source
         */
        SetAxisNames(Manager& manager, XMLReader& source);

        /**
         * Set axes names to one read from tag attribute, same as SetAxisNames(reader.manager, reader.source).
         * @param reader geometry reader
         */
        SetAxisNames(GeometryReader& reader);

        /// Revert axes names to old one.
        ~SetAxisNames() { manager.axisNames = old; }
    };

    explicit Manager(bool draft = false): scriptline(0), axisNames(&AxisNames::axisNamesRegister.get("long, tran, vert")), draft(draft) {}

    virtual ~Manager() {}

    /**
     * Get path hints with given name.
     * @param path_hints_name name of path hints to get
     * @return path hints with given name or @c nullptr if there is no path hints with name @p path_hints_name
     */
    PathHints* getPathHints(const std::string& path_hints_name);

    /**
     * Get path hints with given name.
     * @param path_hints_name name of path hints to get
     * @return path hints with given name or @c nullptr if there is no path hints with name @p path_hints_name
     */
    const PathHints* getPathHints(const std::string& path_hints_name) const;

    /**
     * Get path hints with given name, throw exception if there is no path hints with name @p path_hints_name.
     * @param path_hints_name name of path hints to get
     * @return path hints with given name
     * @throw plask::Exception if there is no path hints with name @p path_hints_name
     */
    PathHints& requirePathHints(const std::string& path_hints_name);

    /**
     * Get path hints with given name, throw exception if there is no path hints with name @p path_hints_name.
     * @param path_hints_name name of path hints to get
     * @return path hints with given name
     * @throw plask::Exception if there is no path hints with name @p path_hints_name
     */
    const PathHints& requirePathHints(const std::string& path_hints_name) const;

    /**
     * Get object with given @p name.
     * @param name name of object
     * @return object with given @p name or @c nullptr if there is no object with given name
     */
    shared_ptr<GeometryObject> getGeometryObject(const std::string& name) const;

    /**
     * Call getObject(name) and try dynamic cast it to @a RequiredObjectType.
     * @param name name of object
     * @return object (casted to RequiredObjectType) with given @p name or @c nullptr if there is no object with given name or object with given name is not of type @a RequiredObjectType
     * @tparm RequiredObjectType required type of object
     */
    template <typename RequiredObjectType>
    shared_ptr<RequiredObjectType> getGeometryObject(const std::string& name) const;

    /**
     * Get object with given name or throw exception if object with given name does not exist.
     * @param name name of object
     * @return object with given name
     * @throw NoSuchGeometryObject if there is no object with given name
     */
    shared_ptr<GeometryObject> requireGeometryObject(const std::string& name);

    /**
     * Call requireElement(name) and try dynamic cast it to @a RequiredObjectType.
     * @param name name of object
     * @return object (casted to RequiredObjectType) with given @p name
     * @tparam RequiredObjectType required type of object
     * @throw UnexpectedGeometryObjectTypeException if requested object is not of type RequiredObjectType
     * @throw NoSuchGeometryObject if there is no object with given name
     */
    template <typename RequiredObjectType>
    shared_ptr<RequiredObjectType> requireGeometryObject(const std::string& name);

    /**
     * Get geometry trunk with given @p name.
     * @param name name of calculation space to get
     * @return calculation space with given @p name or shared_ptr<Geometry>() if there is geometry with given @p name
     */
    shared_ptr<Geometry> getGeometry(const std::string& name) const;

    /**
     * Get mesh with given \p name.
     * \param name name of calculation space to get
     * \return calculation space with given \p name or shared_ptr<Mesh>() if there is no mesh with given @p name
     */
    shared_ptr<MeshBase> getMesh(const std::string& name) const;

    /**
     * Get geometry trunk with given @p name and try dynamic cast it to @a RequiredCalcSpaceType.
     * @param name name of calculation space to get
     * @return required calculation space or shared_ptr<Geometry>() if there is no calculation space with given @p name or can't be casted to RequiredCalcSpaceType
     */
    template <typename RequiredCalcSpaceType>
    shared_ptr<RequiredCalcSpaceType> getGeometry(const std::string& name) const;

    /**
     * Get geometry trunk with given @p name and try dynamic cast it to @a RequiredCalcSpaceType.
     * @param name name of calculation space to get
     * @return required calculation space or shared_ptr<Geometry>() if there is no calculation space with given @p name or can't be casted to RequiredCalcSpaceType
     */
    template <typename RequiredCalcSpaceType>
    shared_ptr<RequiredCalcSpaceType> requireGeometry(const std::string& name) const {
        auto geometry = getGeometry<RequiredCalcSpaceType>(name);
        if (!geometry) throw NoSuchGeometry(name);
        return geometry;
    }

    /**
     * Load constants defintions from the @p reader.
     * @param reader XMLreader to load from, should point to @c \<defines> tag, after read it will be point to @c \</defines> tag
     */
    virtual void loadDefines(XMLReader& reader);

    /**
     * Load geometry using geometry @p reader.
     * @param reader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     */
    virtual void loadGeometry(GeometryReader& reader);

    /**
     * Load materials using @p reader to default database.
     * @param reader reader to read from, should point to @c \<materials> tag, after read it will be point to @c \</materials> tag
     */
    virtual void loadMaterials(XMLReader& reader);

    /**
     * Load meshes and mesh generators using reader.
     * @param reader reader to read from, should point to @c \<solver> tag, after read it will be point to @c \</solver> tag
     */
    virtual void loadGrids(XMLReader& reader);

    /**
     * Load solvers using reader.
     * @param reader reader to read from, should point to @c \<solver> tag, after read it will be point to @c \</solver> tag
     */
    virtual void loadSolvers(XMLReader& reader);

    /**
     * Load solvers intrconnects from the @p reader.
     * \param reader XMLreader to load from
     */
    virtual void loadConnects(XMLReader& reader);

    /**
     * Load script from the @p reader. Do not execute it.
     * \param reader XMLreader to load from
     * \return read script
     */
    virtual void loadScript(XMLReader& reader);

    /**
     * Load geometry using XML reader.
     * @param XMLreader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromReader(XMLReader& XMLreader, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content, will be closed and deleted after read
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromStream(std::unique_ptr<std::istream>&& input, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from string which consist of XML.
     * @param input_XML_str string with XML content
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromXMLString(const std::string &input_XML_str, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /*
     * Read all objects up to end of XML tag and call functor(object) for each object which was read.
     * @param source
     * @param functor
     * @tparam FunctorType unary functor which can take RequiredObjectType& as argument
     * @tparam RequiredObjectType required type of object
     */
    /*template <typename FunctorType, typename RequiredObjectType = GeometryObject>
    void readAllObjects(XMLReader& source, FunctorType functor);*/

    /**
     * Load geometry from XML file.
     * @param fileName name of XML file
     */
    void loadFromFile(const std::string& fileName);

    /**
     * Load geometry from C file object.
     * @param file open file object
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromFILE(FILE* file, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

private:
    /**
     * Get boundary stored in boundaries map. Throw exception if there is no boundary with a given @p name.
     * @param reader
     * @param name name of the boundary to get
     * @return the boundary got
     */
    template <typename Boundary>
    Boundary getBoundaryByName(XMLReader& reader, const std::string& name) {
        auto p = this->boundaries.find(name);
        if (p == this->boundaries.end())
            throw XMLException(reader, format("Can't find boundary (place) with given name \"{0}\".", name));
        return boost::any_cast<Boundary>(p->second);
    }

    /**
     * Store boundary in boundaries map. Throw exception if there is already boundary with a given @p name in the map.
     * @param name name of the boundary to store
     * @param boundary
     */
    void storeBoundary(const std::string& name, boost::any&& boundary) {
        if (!this->boundaries.insert(std::make_pair(name, std::move(boundary))).second)
            throw NamesConflictException("Place (boundary)", name);
    }

public:
    /**
     * Read boundary (place) from current tag and move parser to end of the current tag.
     * @return the boundary read
     */
    template <typename Boundary>
    Boundary readBoundary(XMLReader& reader);

    /**
     * Read boundary conditions from current tag and move parser to end of current tag.
     *
     * Use MeshT static methods to read boundaries, and @c parseBoundaryValue to parse values of conditions:
     * @code
     * template <typename ConditionT> ConditionT parseBoundaryValue(const XMLReader& tag_with_value);
     * @endcode
     * (by default it just read value from "value" attribute)
     *
     * Require format (one or more tag as below):
     * @code
     * \<condition [place="mesh type related place description"] [placename="name of this place"] [placeref="name of earlier stored place"] value attributes read by parseBoundaryValue>
     *  [\<place [name="name of this place"] [mesh-type related]>
     *     ...mesh type related place description...
     *   \</place>|<intersection>|<union>|<difference>]
     * \</condition>
     * @endcode
     * With restrictions:
     * - place must be given exactly once (as attribute or tag), and only in case if placeref was not given;
     * - place name can be given only if placeref was not given;
     * - place name must be unique for all places in XML, and must be given before any placeref which refer to it;
     * - condition value must be in format required by parseBoundaryValue for given type (in most cases it is just one attribute: value).
     * @param reader source of XML data
     * @param dest place to append read conditions
     */
    //TODO moves to modules reader (with names map)
    //@param geometry (optional) geometry used by solver which reads boundary conditions
    template <typename Boundary, typename ConditionT>
    void readBoundaryConditions(XMLReader& reader, BoundaryConditions<Boundary, ConditionT>& dest/*, shared_ptr<Geometry> geometry = shared_ptr<Geometry>()*/);

    /**
     * Load XML content.
     * @param XMLreader XML data source, to load
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     * @param section_filter predicate which returns @c true only if given section should be read, by default it always return @c true
     */
    void load(XMLReader& XMLreader,
              const LoadFunCallbackT& load_from_cb = &disallowExternalSources,
              const std::function<bool(const std::string& section_name)>& section_filter = &acceptAllSections);

    /**
     * Load one section from XML content.
     * @param XMLreader XML data source, to load
     * @param section_to_load name of section to load
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadSection(XMLReader& XMLreader, const std::string& section_to_load,
              const LoadFunCallbackT& load_from_cb = &disallowExternalSources) {
        load(XMLreader, load_from_cb, [&](const std::string& section_name) -> bool { return section_name == section_to_load; });
    }

    /**
     * Try to find mistake in position of objects.
     *
     * Reports (by calling @p callback) when some object has, probably by mistake, different position in two geometries.
     * @param callback call for each pair of geometries in which objects with different (by mistake) positions have been found. Details are passed in parameters:
     *  - 2 geometries of the same type,
     *  - non-empty vector of objects with different positions in the geometries,
     *  - map that allow to obtain name of object or geometry (all object from the passed vector are in this map, but some of the passed geometries can be not).
     */
    void validatePositions(const std::function<void(const Geometry*, const Geometry*, std::vector<const GeometryObject*>&&, const std::map<const GeometryObject*, const char*>&)>& callback) const;

    /**
     * Try to find mistake in position of objects.
     *
     * Reports in returned string when some object has, probably by mistake, different position in two geometries.
     * @return raport, can be multiple-line, empty only if no problems have been found
     */
    void validatePositions() const;

    /**
     * Get index of geometry in root.
     * @param geom geometry to get index
     * @return index of @p geom in root vector, equal to size() if not found
     */
    std::size_t getRootIndex(const Geometry* geom) const;
};

// Specialization for most types
template <typename RequiredObjectType>
shared_ptr<RequiredObjectType> Manager::getGeometryObject(const std::string& name) const {
    return dynamic_pointer_cast<RequiredObjectType>(getGeometryObject(name));
}

// Specialization for GeometryObject which doesn't require dynamic_cast
template <>
inline shared_ptr<GeometryObject> Manager::getGeometryObject<GeometryObject>(const std::string& name) const {
    return getGeometryObject(name);
}

// Specialization for most types
template <typename RequiredObjectType>
inline shared_ptr<RequiredObjectType> Manager::requireGeometryObject(const std::string& name) {
    shared_ptr<RequiredObjectType> result = dynamic_pointer_cast<RequiredObjectType>(requireGeometryObject(name));
    if (!result) throwErrorIfNotDraft(UnexpectedGeometryObjectTypeException());
    return result;
}

// Specialization for GeometryObject which doesn't require dynamic_cast
template <>
inline shared_ptr<GeometryObject> Manager::requireGeometryObject<GeometryObject>(const std::string& name) {
    return requireGeometryObject(name);
}

//specialization for most types
template <typename RequiredCalcSpaceType>
inline shared_ptr<RequiredCalcSpaceType> Manager::getGeometry(const std::string& name) const {
    return dynamic_pointer_cast<RequiredCalcSpaceType>(getGeometry(name));
}

// Specialization for Geometry which doesn't require dynamic_cast
template <>
inline shared_ptr<Geometry> Manager::getGeometry<Geometry>(const std::string& name) const {
    return getGeometry(name);
}

/**
 * Parse condition from XML tag.
 *
 * Default implementation just read it from "value" attribute which use boost::lexical_cast to convert it to given type @p ConditionT.
 * Specializations for choosen types can require other attributes.
 * @param tag_with_value XML tag to parse
 * @return parsed condition
 * @tparam ConditionT type of condition to parse
 */
template <typename ConditionT>
inline ConditionT parseBoundaryValue(const XMLReader& tag_with_value) {
    return tag_with_value.requireAttribute<ConditionT>("value");
}

template <typename Boundary>
Boundary Manager::readBoundary(XMLReader& reader) {
    Boundary boundary;
    std::string op_name = reader.getTagName();
    plask::optional<std::string> placename = reader.getAttribute("name");
    if (op_name == "union") {
        reader.requireTag(); Boundary A = this->readBoundary<Boundary>(reader);
        reader.requireTag(); Boundary B = this->readBoundary<Boundary>(reader);
        reader.requireTagEnd();
        boundary = A + B;
    } else
    if (op_name == "intersection") {
        reader.requireTag(); Boundary A = this->readBoundary<Boundary>(reader);
        reader.requireTag(); Boundary B = this->readBoundary<Boundary>(reader);
        reader.requireTagEnd();
        boundary = A * B;
    } else
    if (op_name == "difference") {
        reader.requireTag(); Boundary A = this->readBoundary<Boundary>(reader);
        reader.requireTag(); Boundary B = this->readBoundary<Boundary>(reader);
        reader.requireTagEnd();
        boundary = A - B;
    } else
    if (op_name == "place") {
        reader.ensureNodeTypeIs(XMLReader::NODE_ELEMENT, "place");
        plask::optional<std::string> refname = reader.getAttribute("ref");
        boundary = refname ? this->getBoundaryByName<Boundary>(reader, *refname)
                           : parseBoundary<Boundary>(reader, *this);
    } else
        reader.throwUnexpectedElementException("place, union, intersection, or difference tag");
    if (boundary.isNull()) throwErrorIfNotDraft(XMLException(reader, "Can't parse boundary place from XML."));
    if (placename) {
        std::replace(placename->begin(), placename->end(), '-', '_');
        this->storeBoundary(*placename, boundary);
    }
    return boundary;
}

template <typename Boundary, typename ConditionT>
inline void Manager::readBoundaryConditions(XMLReader& reader, BoundaryConditions<Boundary, ConditionT>& dest) {
    while (reader.requireTagOrEnd("condition")) {
        Boundary boundary;
        plask::optional<std::string> place = reader.getAttribute("place");
        plask::optional<std::string> placename = reader.getAttribute("placename");
        ConditionT value;
        try {
            value = parseBoundaryValue<ConditionT>(reader);
        } catch (std::runtime_error err) {
            throwErrorIfNotDraft(err);
        }
        if (place) {
            boundary = parseBoundary<Boundary>(*place, *this);
            if (boundary.isNull()) throwErrorIfNotDraft(XMLException(reader, format("Can't parse boundary place from string \"{0}\".", *place)));
        } else {
            place = reader.getAttribute("placeref");
            if (place)
                boundary = this->getBoundaryByName<Boundary>(reader, *place);
            else {
                reader.requireTag();
                boundary = this->readBoundary<Boundary>(reader);
                //placename.reset(); // accept "placename" or only "name" in place tag?
            }
        }
        /*if (!value) {   // value still not known, must be read from tag <value>...</value>
            reader.requireTag("value");
            *value = reader.requireText<ConditionT>();
            reader.requireTagEnd();
        }*/ //now we read only from XML tags
        if (placename) this->storeBoundary(*placename, boundary);
        dest.add(std::move(boundary), std::move(value));
        reader.requireTagEnd(); // </condition>
    }
}


} // namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
