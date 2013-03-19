#include "solver.h"

#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file includes:
- plask::Manager class.
*/

#include <string>
#include <map>
#include <set>
#include <boost/filesystem.hpp>

#include "utils/xml/reader.h"

#include "mesh/mesh.h"
#include "material/db.h"
#include "geometry/path.h"
#include "geometry/space.h"
#include "geometry/reader.h"

namespace plask {

class Solver;

/**
 * Geometry manager features:
 * - read/write geometries,
 * - allow for access to geometry objects (also by names).
 *
 * @see @ref geometry
 */
struct Manager {

    /// Throw exception with information that loading from external sources is not supported or disallowed.
    static void disallowExternalSources(Manager& manager, const MaterialsSource& materialsSource, const std::string& url, const std::string& section) {
        throw Exception("Can't load section \"%1%\" from \"%2%\". Loading from external sources is not supported or disallowed.", section, url); }

    /// Allow to support reading some sections from other files.
    struct ExternalSourcesFromFile {

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
         * Create ExternalSourcesFromFile which dosn't support relative file names
         * (original file name is not known).
         */
        ExternalSourcesFromFile(): prev(nullptr) {}

        /**
         * Create ExternalSourcesFromFile which support relative file names.
         * @param originalFileName name of file from which XML is read now
         */
        ExternalSourcesFromFile(const boost::filesystem::path& originalFileName,
                                const std::string& currentSection = std::string(),
                                ExternalSourcesFromFile* prev = nullptr)
            : originalFileName(originalFileName), currentSection(currentSection), prev(prev) {}

        void operator()(Manager& manager, const MaterialsSource& materialsSource, const std::string& url, const std::string& section);

    };

private:

    /// Map holding information about global solvers names stored in \p solvers.lst
    std::map<std::string, std::map<std::string, std::string>> global_solver_names;

    /// @return @c true
    static bool acceptAllSections(const std::string&) { return true; }

    /*struct LoadFunCallbackT {
        std::pair< XMLReader, std::unique_ptr<LoadFunCallbackT> > get(const std::string& url) const {
            throw Exception("Can't load section from \"%1%\". Loading from external sources is not supported or disallowed.", url);
        }
    };*/

    /// Load section from external url, throw excpetion in case of errors, takes as parameters: manager, materials source, url and section name.
    typedef std::function<void(Manager&, const MaterialsSource&, const std::string&, const std::string&)> LoadFunCallbackT;

    /**
     * Try load section from external location.
     *
     * Check if current tag in @p reader has "from" attribute, and if has load section represented by this tag from external location using @p load_from.
     * @param reader XML data source
     * @param materialsSource source of materials
     * @param load_from
     */
    bool tryLoadFromExternal(XMLReader& reader, const MaterialsSource& materialsSource, const LoadFunCallbackT& load_from);

  protected:

    /**
     * Load solver from file.
     * @param category, lib, solver_name solver parameters
     * @return loaded solver
     */
    virtual shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name);

  public:

    static constexpr const char* TAG_NAME_ROOT = "plask";           ///< name of root XML tag
    static constexpr const char* TAG_NAME_DEFINES = "defines";      ///< name of XML tag of section with const definitions
    static constexpr const char* TAG_NAME_MATERIALS = "materials";  ///< name of XML tag of section with materials
    static constexpr const char* TAG_NAME_GEOMETRY = "geometry";    ///< name of XML tag of section with geometry
    static constexpr const char* TAG_NAME_GRIDS = "grids";          ///< name of XML tag of section with grids (meshes and generators)
    static constexpr const char* TAG_NAME_SOLVERS = "solvers";       ///< name of XML tag of section with solvers
    static constexpr const char* TAG_NAME_CONNECTS = "connects";    ///< name of XML tag of section with provides-receiver connections
    static constexpr const char* TAG_NAME_SCRIPT = "script";        ///< name of XML tag of section with (python) script

    /// Allow to access path hints by name.
    std::map<std::string, PathHints> pathHints;

    /// Allow to access objects by name.
    std::map<std::string, shared_ptr<GeometryObject> > namedObjects;

    /// Roots objects, geometries.
    std::vector< shared_ptr<Geometry> > roots;

    /// Geometries (calculation spaces) by name.
    std::map<std::string, shared_ptr<Geometry> > geometries;

    /// Meshes by name.
    std::map< std::string, shared_ptr<Mesh> > meshes;

    /// Meshes generators by name.
    std::map< std::string, shared_ptr<MeshGenerator> > generators;

    /// Solvers by name.
    std::map< std::string, shared_ptr<Solver> > solvers;

    /// Boundaries places by name.
    //TODO? move to special modules reader class to have more local scope?
    std::map< std::string, boost::any > boundaries;

    /// Script read from file
    std::string script;

    /// Line in which script begins
    unsigned scriptline;

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
     * @tparam RequiredObjectType required type of object
     */
    template <typename RequiredObjectType>
    shared_ptr<RequiredObjectType> getGeometryObject(const std::string& name) const;

    /**
     * Get object with given name or throw exception if object with given name does not exist.
     * @param name name of object
     * @return object with given name
     * @throw NoSuchGeometryObject if there is no object with given name
     */
    shared_ptr<GeometryObject> requireGeometryObject(const std::string& name) const;

    /**
     * Call requireElement(name) and try dynamic cast it to @a RequiredObjectType.
     * @param name name of object
     * @return object (casted to RequiredObjectType) with given @p name
     * @tparam RequiredObjectType required type of object
     * @throw UnexpectedGeometryObjectTypeException if requested object is not of type RequiredObjectType
     * @throw NoSuchGeometryObject if there is no object with given name
     */
    template <typename RequiredObjectType>
    shared_ptr<RequiredObjectType> requireGeometryObject(const std::string& name) const;

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
    shared_ptr<Mesh> getMesh(const std::string& name) const;

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
     * Load constants defintions from the file.
     * \param reader XMLreader to load from
     */
    virtual void loadDefines(XMLReader& reader);

    /**
     * Load geometry using geometry reader.
     * @param reader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     */
    void loadGeometry(GeometryReader& reader);

    /**
     * Load materials using geometry reader.
     * @param reader reader to read from, should point to @c \<materials> tag, after read it will be point to @c \</materials> tag
     * @param materialsSource materials source, which was passed to load method
     *  (in case of using material database, you can convert @p materialsSource back to MaterialsDB using Manager::getMaterialsDBfromSource)
     */
    virtual void loadMaterials(XMLReader& reader, const MaterialsSource& materialsSource);

    /**
     * Load meshes and mesh generators using reader.
     * @param reader reader to read from, should point to @c \<solver> tag, after read it will be point to @c \</solver> tag
     */
    void loadGrids(XMLReader& reader);

    /**
     * Load solvers using reader.
     * @param reader reader to read from, should point to @c \<solver> tag, after read it will be point to @c \</solver> tag
     */
    void loadSolvers(XMLReader& greader);

    /**
     * Load solvers intrconnects from the file.
     * \param reader XMLreader to load from
     */
    virtual void loadConnects(XMLReader& reader);

    /**
     * Load script from the file. Do not execute it.
     * \param reader XMLreader to load from
     * \return read script
     */
    void loadScript(XMLReader& reader);

    /**
     * Load geometry using XML reader.
     * @param XMLreader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     * @param materialsDB materials database, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromReader(XMLReader& XMLreader, const MaterialsDB& materialsDB = MaterialsDB::getDefault(), const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry using XML reader.
     * @param XMLreader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     * @param materialsSource source of materials, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromReader(XMLReader& XMLreader, const MaterialsSource& materialsSource, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content, will be closed and deleted after read
     * @param materialsDB materials database, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromStream(std::istream* input, const MaterialsDB& materialsDB = MaterialsDB::getDefault(), const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content, will be closed and deleted after read
     * @param materialsSource source of materials, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromStream(std::istream* input, const MaterialsSource& materialsSource, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from string which consist of XML.
     * @param input_XML_str string with XML content
     * @param materialsDB materials database, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB = MaterialsDB::getDefault(), const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from string which consist of XML.
     * @param input_XML_str string with XML content
     * @param materialsSource source of materials, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromXMLString(const std::string &input_XML_str, const MaterialsSource& materialsSource, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

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
     * @param materialsDB materials database, used to get materials by name for leafs
     */
    void loadFromFile(const std::string& fileName, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from XML file.
     * @param fileName name of XML file
     * @param materialsSource source of materials, used to get materials by name for leafs
     */
    void loadFromFile(const std::string& fileName, const MaterialsSource& materialsSource);

    /**
     * Load geometry from C file object.
     * @param file open file object
     * @param materialsDB materials database, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromFILE(FILE* file, const MaterialsDB& materialsDB = MaterialsDB::getDefault(), const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

    /**
     * Load geometry from C file object.
     * @param file open file object
     * @param materialsSource source of materials, used to get materials by name for leafs
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadFromFILE(FILE* file, const MaterialsSource& materialsSource, const LoadFunCallbackT& load_from_cb = &disallowExternalSources);

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
     *   \</place>]
     * \</condition>
     * @endcode
     * With restrictions:
     * - place must be given exactly once (as attribute or tag), and only in case if placeref was not given;
     * - place name can be given only if placeref was not given;
     * - place name must be unique for all places in XML, and must be given before any placeref which refer to it;
     * - condition value must be in format required by parseBoundaryValue for given type (in most cases it is just one attribute: value).
     * @param reader source of XML data
     * @param geometry (optional) geometry used by solver which reads boundary conditions
     * @param dest place to append read conditions
     */
    //TODO moves to modules reader (with names map)
    template <typename MeshT, typename ConditionT>
    void readBoundaryConditions(XMLReader& reader, BoundaryConditions<MeshT, ConditionT>& dest, shared_ptr<Geometry> geometry = shared_ptr<Geometry>());

    /**
     * Load XML content.
     * @param XMLreader XML data source, to load
     * @param materialsSource source of materials, typically materials database
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     * @param section_filter predicate which returns @c true only if given section should be read, by default it always return @c true
     */
    void load(XMLReader& XMLreader, const MaterialsSource& materialsSource,
              const LoadFunCallbackT& load_from_cb = &disallowExternalSources,
              const std::function<bool(const std::string& section_name)>& section_filter = &acceptAllSections);

    /**
     * Load one section from XML content.
     * @param XMLreader XML data source, to load
     * @param section_to_load name of section to load
     * @param materialsSource source of materials, typically materials database
     * @param load_from_cb callback called to open external location, allow loading some section from another sources,
     *  this callback should read section from external XML source pointed by url (typically name of file) or throw exception
     */
    void loadSection(XMLReader& XMLreader, const std::string& section_to_load, const MaterialsSource& materialsSource,
              const LoadFunCallbackT& load_from_cb = &disallowExternalSources) {
        load(XMLreader, materialsSource, load_from_cb, [&](const std::string& section_name) -> bool { return section_name == section_to_load; });
    }

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

//specialization for most types
template <typename RequiredObjectType>
inline shared_ptr<RequiredObjectType> Manager::requireGeometryObject(const std::string& name) const {
    shared_ptr<RequiredObjectType> result = dynamic_pointer_cast<RequiredObjectType>(requireGeometryObject(name));
    if (!result) throw UnexpectedGeometryObjectTypeException();
    return result;
}

// Specialization for GeometryObject which doesn't require dynamic_cast
template <>
inline shared_ptr<GeometryObject> Manager::requireGeometryObject<GeometryObject>(const std::string& name) const {
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

template <typename MeshT, typename ConditionT>
inline void Manager::readBoundaryConditions(XMLReader& reader, BoundaryConditions<MeshT, ConditionT>& dest, shared_ptr<Geometry> geometry) {
    BoundaryParserEnviroment parser_enviroment(*this, geometry);
    while (reader.requireTagOrEnd("condition")) {
        Boundary<MeshT> boundary;
        boost::optional<std::string> place = reader.getAttribute("place");
        boost::optional<std::string> placename = reader.getAttribute("placename");
        //boost::optional<ConditionT> value = reader.getAttribute<ConditionT>("value");
        ConditionT value = parseBoundaryValue<ConditionT>(reader);
        if (place) {
            boundary = parseBoundary<MeshT>(*place, parser_enviroment);
            if (boundary.isNull()) throw XMLException(reader, format("Can't parse boundary place from string \"%1%\".", *place));
        } else {
            place = reader.getAttribute("placeref");
            if (place) {
                auto p = this->boundaries.find(*place);
                if (p == this->boundaries.end())
                    throw XMLException(reader, format("Can't find boundary (place) with given name \"%1%\".", *place));
                boundary = boost::any_cast<Boundary<MeshT>>(p->second);
            } else {
                reader.requireTag("place");
                if (!placename) placename = reader.getAttribute("name");
                boundary = parseBoundary<MeshT>(reader, parser_enviroment);
                if (boundary.isNull()) throw XMLException(reader, "Can't parse boundary place from XML.");
            }
        }
        /*if (!value) {   // value still not known, must be read from tag <value>...</value>
            reader.requireTag("value");
            *value = reader.requireText<ConditionT>();
            reader.requireTagEnd();
        }*/ //now we read only from XML tags
        if (placename) {
            if (!this->boundaries.insert(std::make_pair(*placename, boost::any(boundary))).second)
                throw NamesConflictException("Place (boundary)", *placename);
        }
        dest.add(std::move(boundary), /*std::move(*value)*/ std::move(value));
        reader.requireTagEnd(); //</condition>
    }
}


} // namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
