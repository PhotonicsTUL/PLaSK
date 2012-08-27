#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file includes:
- plask::Manager class.
*/

#include <string>
#include <map>
#include <set>

#include "utils/xml/reader.h"

#include "mesh/mesh.h"
#include "material/db.h"
#include "geometry/path.h"
#include "geometry/space.h"
#include "geometry/reader.h"

#include "solver.h"

namespace plask {

/**
 * Geometry manager features:
 * - read/write geometries,
 * - allow for access to geometry elements (also by names).
 *
 * @see @ref geometry
 */
class Manager {

    template <typename MaterialsSource>
    void load(XMLReader& XMLreader, const MaterialsSource& materialsSource);

  protected:

    /**
     * Load solver from file.
     * @param category, lib, solver_name solver parameters
     * @return loaded solver
     */
    virtual shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name);

  public:
    
    static constexpr const char* TAG_NAME_ROOT = "plask";           ///< name of root XML tag
    static constexpr const char* TAG_NAME_MATERIALS = "materials";  ///< name of XML tag of section with materials
    static constexpr const char* TAG_NAME_GEOMETRY = "geometry";    ///< name of XML tag of section with geometry
    static constexpr const char* TAG_NAME_GRIDS = "grids";          ///< name of XML tag of section with grids (meshes and generators)
    static constexpr const char* TAG_NAME_SOVERS = "solvers";       ///< name of XML tag of section with solvers
    static constexpr const char* TAG_NAME_CONNECTS = "connects";    ///< name of XML tag of section with provides-receiver connections
    static constexpr const char* TAG_NAME_SCRIPT = "script";        ///< name of XML tag of section with (python) script

    /// Allow to access path hints by name.
    std::map<std::string, PathHints> pathHints;

    /// Allow to access elements by name.
    std::map<std::string, shared_ptr<GeometryElement> > namedElements;

    /// Roots elements, geometries.
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
     * Get element with given @p name.
     * @param name name of element
     * @return element with given @p name or @c nullptr if there is no element with given name
     */
    shared_ptr<GeometryElement> getGeometryElement(const std::string& name) const;

    /**
     * Call getElement(name) and try dynamic cast it to @a RequiredElementType.
     * @param name name of element
     * @return element (casted to RequiredElementType) with given @p name or @c nullptr if there is no element with given name or element with given name is not of type @a RequiredElementType
     * @tparam RequiredElementType required type of element
     */
    template <typename RequiredElementType>
    shared_ptr<RequiredElementType> getGeometryElement(const std::string& name) const;

    /**
     * Get element with given name or throw exception if element with given name does not exist.
     * @param name name of element
     * @return element with given name
     * @throw NoSuchGeometryElement if there is no element with given name
     */
    shared_ptr<GeometryElement> requireGeometryElement(const std::string& name) const;

    /**
     * Call requireElement(name) and try dynamic cast it to @a RequiredElementType.
     * @param name name of element
     * @return element (casted to RequiredElementType) with given @p name
     * @tparam RequiredElementType required type of element
     * @throw UnexpectedGeometryElementTypeException if requested element is not of type RequiredElementType
     * @throw NoSuchGeometryElement if there is no element with given name
     */
    template <typename RequiredElementType>
    shared_ptr<RequiredElementType> requireGeometryElement(const std::string& name) const;

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
     * Load geometry using geometry reader.
     * @param reader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     */
    void loadGeometry(GeometryReader& reader);

    /**
     * Load materials using geometry reader.
     * \param reader reader to read from, should point to @c \<materials> tag, after read it will be point to @c \</materials> tag
     * \param materialsDB materials database
     */
    virtual void loadMaterials(XMLReader& reader, MaterialsDB& materialsDB = MaterialsDB::getDefault());

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
     */
    void loadFromReader(XMLReader& XMLreader, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry using XML reader.
     * @param XMLreader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     * @param materialsSource source of materials, used to get materials by name for leafs
     */
    void loadFromReader(XMLReader& XMLreader, const GeometryReader::MaterialsSource& materialsSource);

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content
     * @param materialsDB materials database, used to get materials by name for leafs
     */
    void loadFromStream(std::istream &input, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content
     * @param materialsSource source of materials, used to get materials by name for leafs
     */
    void loadFromStream(std::istream &input, const GeometryReader::MaterialsSource& materialsSource);

    /**
     * Load geometry from string which consist of XML.
     * @param input_XML_str string with XML content
     * @param materialsDB materials database, used to get materials by name for leafs
     */
    void loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from string which consist of XML.
     * @param input_XML_str string with XML content
     * @param materialsSource source of materials, used to get materials by name for leafs
     */
    void loadFromXMLString(const std::string &input_XML_str, const GeometryReader::MaterialsSource& materialsSource);

    /*
     * Read all elements up to end of XML tag and call functor(element) for each element which was read.
     * @param source
     * @param functor
     * @tparam FunctorType unary functor which can take RequiredElementType& as argument
     * @tparam RequiredElementType required type of element
     */
    /*template <typename FunctorType, typename RequiredElementType = GeometryElement>
    void readAllElements(XMLReader& source, FunctorType functor);*/

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
    void loadFromFile(const std::string& fileName, const GeometryReader::MaterialsSource& materialsSource);

    /**
     * Load geometry from C file object.
     * @param file open file object
     * @param materialsDB materials database, used to get materials by name for leafs
     */
    void loadFromFILE(FILE* file, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from C file object.
     * @param file open file object
     * @param materialsSource source of materials, used to get materials by name for leafs
     */
    void loadFromFILE(FILE* file, const GeometryReader::MaterialsSource& materialsSource);

    /**
     * Read boundary conditions from current tag and move parser to end of current tag.
     *
     * Use MeshT static methods to read boundaries, and boost::lexical_cast<ConditionT> to parse values of conditions.
     *
     * Require format (one or more tag as below):
     * @code
     * \<condition [place="mesh type related place description"] [placename="name of this place"] [placeref="name of earlier stored place"] [value="condition value"]>
     *  [\<place [name="name of this place"] [mesh-type related]>
     *     ...mesh type related place description...
     *   \</place>]
     *  [\<value>condition value\</value>]
     * \</condition>
     * @endcode
     * With restrictions:
     * - place must be given exactly once (as attribute or tag), and only in case if placeref was not given;
     * - place name can be given only if placeref was not given;
     * - place name must be unique for all places in XML, and must be given before any placeref which refer to it;
     * - condition value must be given exaclty once (as attribute or in value tag).
     * @param reader source of XML data
     * @param dest place to append read conditions
     */
    //TODO moves to modules reader (with names map)
    template <typename MeshT, typename ConditionT>
    void readBoundaryConditions(XMLReader& reader, BoundaryConditions<MeshT, ConditionT>& dest);
};

// Specialization for most types
template <typename RequiredElementType>
shared_ptr<RequiredElementType> Manager::getGeometryElement(const std::string& name) const {
    return dynamic_pointer_cast<RequiredElementType>(getGeometryElement(name));
}

// Specialization for GeometryElement which doesn't require dynamic_cast
template <>
inline shared_ptr<GeometryElement> Manager::getGeometryElement<GeometryElement>(const std::string& name) const {
    return getGeometryElement(name);
}

//specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> Manager::requireGeometryElement(const std::string& name) const {
    shared_ptr<RequiredElementType> result = dynamic_pointer_cast<RequiredElementType>(requireGeometryElement(name));
    if (!result) throw UnexpectedGeometryElementTypeException();
    return result;
}

// Specialization for GeometryElement which doesn't require dynamic_cast
template <>
inline shared_ptr<GeometryElement> Manager::requireGeometryElement<GeometryElement>(const std::string& name) const {
    return requireGeometryElement(name);
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

template <typename MeshT, typename ConditionT>
inline void Manager::readBoundaryConditions(XMLReader& reader, BoundaryConditions<MeshT, ConditionT>& dest) {
    while (reader.requireTagOrEnd("condition")) {
        Boundary<MeshT> boundary;
        boost::optional<std::string> place = reader.getAttribute("place");
        boost::optional<std::string> placename = reader.getAttribute("placename");
        boost::optional<ConditionT> value = reader.getAttribute<ConditionT>("value");
        if (place) {
            boundary = parseBoundary<MeshT>(*place);
            if (boundary.isNull()) throw Exception("Can't parse boundary place from string \"%1%\".", *place);
        } else {
            place = reader.getAttribute("placeref");
            if (place) {
                auto p = this->boundaries.find(*place);
                if (p == this->boundaries.end())
                    throw Exception("Can't find boundary (place) with given name \"%1%\".", *place);
                boundary = boost::any_cast<Boundary<MeshT>>(p->second);
            } else {
                reader.requireTag("place");
                if (!placename) placename = reader.getAttribute("name");
                boundary = parseBoundary<MeshT>(reader);
                if (boundary.isNull()) throw Exception("Can't parse boundary place from XML.", *place);
            }
        }
        if (!value) {   // value still not known, must be read from tag <value>...</value>
            reader.requireTag("value");
            *value = reader.requireText<ConditionT>();
            reader.requireTagEnd();
        }
        if (placename) {
            if (!this->boundaries.insert(std::make_pair(*placename, boost::any(boundary))).second)
                throw NamesConflictException("Place (boundary)", *placename);
        }
        dest.add(std::move(boundary), std::move(*value));
        reader.requireTagEnd(); //</condition>
    }
}


template <typename SpaceT>
void SolverOver<SpaceT>::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "geometry") {
            auto name = reader.getAttribute("ref");
            if (!name) name.reset(reader.requireText());
            auto found = manager.geometries.find(*name);
            if (found == manager.geometries.end())
                throw BadInput(this->getId(), "Geometry '%1%' not found.", *name);
            else {
                auto geometry = dynamic_pointer_cast<SpaceT>(found->second);
                if (!geometry) throw BadInput(this->getId(), "Geometry '%1%' of wrong type.");
                this->setGeometry(geometry);
            }
        } else {
            this->loadParam(reader.getNodeName(), reader, manager);
        }
        reader.requireTagEnd();
    }
}

template <typename SpaceT, typename MeshT>
void SolverWithMesh<SpaceT,MeshT>::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "geometry") {
            auto name = reader.getAttribute("ref");
            if (!name) name.reset(reader.requireText());
            auto found = manager.geometries.find(*name);
            if (found == manager.geometries.end())
                throw BadInput(this->getId(), "Geometry '%1%' not found.", *name);
            else {
                auto geometry = dynamic_pointer_cast<SpaceT>(found->second);
                if (!geometry) throw BadInput(this->getId(), "Geometry '%1%' of wrong type.");
                this->setGeometry(geometry);
            }
        }
        else if (reader.getNodeName() == "mesh") {
            auto name = reader.getAttribute("ref");
            auto found = manager.meshes.find(*name);
            if (found != manager.meshes.end()) {
                auto mesh = dynamic_pointer_cast<MeshT>(found->second);
                if (!mesh) throw BadInput(this->getId(), "Mesh '%1%' of wrong type.");
                this->setMesh(mesh);
            }
            else {
                auto found = manager.generators.find(*name);
                if (found != manager.generators.end()) {
                    auto generator = dynamic_pointer_cast<MeshGeneratorOf<MeshT>>(found->second);
                    if (!generator) throw BadInput(this->getId(), "Mesh '%1%' of wrong type.");
                    this->setMesh(generator);
                } else
                    throw BadInput(this->getId(), "Neither mesh nor mesh generator '%1%' found.", *name);
            }
        } else {
            this->loadParam(reader.getNodeName(), reader, manager);
        }
        reader.requireTagEnd();
    }
}


} // namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
