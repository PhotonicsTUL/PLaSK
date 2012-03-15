#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file includes:
- plask::GeometryManager class.
*/

#include <string>
#include <map>
#include <set>

#include "../utils/xml.h"

#include "../material/db.h"
#include "path.h"

namespace plask {

/**
 * Geometry manager features:
 * - read/write geometries,
 * - allow for access to geometry elements (also by names).
 *
 * @see @ref geometry
 */
struct GeometryManager {

    // Store pointers to all elements.
    //std::set<GeometryElement*> elements;

    /// Allow to access path hints by name.
    std::map<std::string, PathHints> pathHints;

    /// Allow to access elements by name.
    std::map<std::string, weak_ptr<GeometryElement> > namedElements;

    /// Roots elements.
    std::vector< shared_ptr<GeometryElement> > roots;

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
    shared_ptr<GeometryElement> getElement(const std::string& name) const;

    /**
     * Call getElement(name) and try dynamic cast it to @a RequiredElementType.
     * @param name name of element
     * @return element (casted to RequiredElementType) with given @p name or @c nullptr if there is no element with given name or element with given name is not of type @a RequiredElementType
     * @tparam RequiredElementType required type of element
     */
    template <typename RequiredElementType>
    shared_ptr<RequiredElementType> getElement(const std::string& name) const;

    shared_ptr<GeometryElement> getRootElement(const std::size_t index) const { return roots[index]; }

    template <typename RequiredElementType>
    shared_ptr<RequiredElementType> getRootElement(const std::size_t index) const;

    /**
     * Get element with given name or throw exception if element with given name does not exist.
     * @param name name of element
     * @return element with given name
     * @throw NoSuchGeometryElement if there is no element with given name
     */
    shared_ptr<GeometryElement> requireElement(const std::string& name) const;

    /**
     * Call requireElement(name) and try dynamic cast it to @a RequiredElementType.
     * @param name name of element
     * @return element (casted to RequiredElementType) with given @p name
     * @tparam RequiredElementType required type of element
     * @throw UnexpectedGeometryElementTypeException if requested element is not of type RequiredElementType
     * @throw NoSuchGeometryElement if there is no element with given name
     */
    template <typename RequiredElementType>
    shared_ptr<RequiredElementType> requireElement(const std::string& name) const;

    /**
     * Load geometry using XML reader.
     * @param XMLreader reader to read from, should point to <geometry> tag, after read it will be point to </geometry> tag
     */
    void loadFromReader(XMLReader& XMLreader, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from (XML) stream.
     * @param input stream to read from, with XML content
     */
    void loadFromXMLStream(std::istream &input, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Load geometry from string which consist of XML.
     * @param input_XML_str string with XML content
     */
    void loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

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
     */
    void loadFromFile(const std::string& fileName, const MaterialsDB& materialsDB = MaterialsDB::getDefault());
};

//specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryManager::getElement(const std::string& name) const {
    return dynamic_pointer_cast<RequiredElementType>(getElement(name));
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryElement> GeometryManager::getElement<GeometryElement>(const std::string& name) const {
    return getElement(name);
}

//specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryManager::getRootElement(const std::size_t index) const {
    return dynamic_pointer_cast<RequiredElementType>(getRootElement(index));
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryElement> GeometryManager::getRootElement<GeometryElement>(const std::size_t index) const {
    return getRootElement(index);
}

//specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryManager::requireElement(const std::string& name) const {
    shared_ptr<RequiredElementType> result = dynamic_pointer_cast<RequiredElementType>(requireElement(name));
    if (!result) throw UnexpectedGeometryElementTypeException();
    return result;
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryElement> GeometryManager::requireElement<GeometryElement>(const std::string& name) const {
    return requireElement(name);
}

}	// namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
