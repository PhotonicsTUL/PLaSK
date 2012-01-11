#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file includes:
- plask::GeometryManager class.
*/

#include <string>
#include <map>
#include <set>

#include "container.h"

#include "../utils/xml.h"

#include "../material/material.h"

namespace plask {

/**
 * Geometry manager features:
 * - read/write geometries
 * - reserve and free memory needed by geometry structure
 * - allow access to geometry elements (also by names)
 */
struct GeometryManager {

    // Store pointers to all elements.
    //std::set<GeometryElement*> elements;

    /// Allow to access path hints by name.
    std::map<std::string, PathHints> pathHints;

    /// Allow to access elements by name.
    std::map<std::string, shared_ptr<GeometryElement> > namedElements;
    
    /// Material database used by geometry (leafs).
    MaterialsDB& materialsDB;

    /**
     * @param materialsDB material database used by geometry (leafs)
     */
    GeometryManager(MaterialsDB& materialsDB);

    ///Delete all elements.
    ~GeometryManager();
    
    /**
     * Get element with given name.
     * @param name name of element
     * @return element with given name or nullptr if there is no element with given name
     */
    shared_ptr<GeometryElement> getElement(const std::string& name);
    
    /**
     * Get element with given name or throw exception if element with given name does not exist.
     * @param name name of element
     * @return element with given name
     * @throw NoSuchGeometryElement if there is no element with given name
     */
    shared_ptr<GeometryElement> requireElement(const std::string& name);
    
    void loadFromReader(XMLReader& XMLreader);
    
    void loadFromXMLStream(std::istream &input);
    
    void loadFromXMLString(const std::string &input_XML_str);
    
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
    void loadFromFile(const std::string& fileName);
};

}	// namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
