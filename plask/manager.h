#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file includes:
- plask::Manager class.
*/

#include <string>
#include <map>
#include <set>

#include "utils/xml.h"

#include "mesh/mesh.h"
#include "material/db.h"
#include "geometry/path.h"
#include "geometry/space.h"
#include "geometry/reader.h"

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

  public:

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

    /// Meshe generators by name.
    std::map< std::string, shared_ptr<MeshGenerator> > generators;

    //TODO modules map
    //TODO boundaries map (Boundary top class - probably empty, with virtual destructor)

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
     * Load geometry using geometry reader.
     * @param reader reader to read from, should point to @c \<geometry> tag, after read it will be point to @c \</geometry> tag
     */
    void loadGeometry(GeometryReader& reader);

    /**
     * Load meshes and mesh generators using reader.
     * @param reader reader to read from, should point to @c \<module> tag, after read it will be point to @c \</module> tag
     */
    void loadGrids(XMLReader& reader);

    /**
     * Load modules using reader.
     * @param reader reader to read from, should point to @c \<module> tag, after read it will be point to @c \</module> tag
     */
    void loadModules(XMLReader& reader);

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
    void loadFromXMLStream(std::istream &input, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content
     * @param materialsSource source of materials, used to get materials by name for leafs
     */
    void loadFromXMLStream(std::istream &input, const GeometryReader::MaterialsSource& materialsSource);

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

}	// namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
