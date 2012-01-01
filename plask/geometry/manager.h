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

namespace plask {

/**
 * Geometry manager features:
 * - read/write geometries
 * - reserve and free memory needed by geometry structure
 * - allow access to geometry elements (also by names)
 */
struct GeometryManager {
    
    /**
     * Create new geometry element (using new operator) with parameters reading from XML source.
     * Can call managers methods to read children (GeometryManager::readElement).
     * Should throw excpetion if can't create element.
     * Result will be delete (using delete operator) by caller.
     */
    typedef GeometryElement* element_read_f(GeometryManager& manager, XMLReader& source);
    
    /**
     * Global elements readers register.
     *
     * Map: xml tag name -> element reader function.
     */
    static std::map<std::string, element_read_f*> elementReaders;
    
    /**
     * Add reader to elementReaders.
     * @param tag_name XML tag name
     * @param reader element reader function
     */ 
    static void registerElementReader(const std::string& tag_name, element_read_f* reader);
    
    /**
     * Helper which call registerElementReader in constructor.
     *
     * Each element can create one global instanse of this class to register own reader.
     */
    struct RegisterElementReader {
        RegisterElementReader(const std::string& tag_name, element_read_f reader) {
            GeometryManager::registerElementReader(tag_name, reader);   
        }
    };

	/// Store pointers to all elements.
	std::set<GeometryElement*> elements;

	/// Allow to access path hints by name.
	std::map<std::string, PathHints*> pathHints;

	/// Allow to access elements by name.
	std::map<std::string, GeometryElement*> namedElements;

    GeometryManager();

    ///Delete all elements.
    ~GeometryManager();
    
    /**
     * Get element with given name.
     * @param name name of element
     * @return element with given name or nullptr if there is no element with given name
     */
    GeometryElement* getElement(const std::string& name);
    
    /**
     * Get element with given name or throw exception if element with given name does not exist.
     * @param name name of element
     * @return element with given name
     * @throw NoSuchGeometryElement if there is no element with given name
     */
    GeometryElement& requireElement(const std::string& name);
    
    /**
     * Read geometry element from @a source and add it GeometryManager structures.
     *
     * Typically it creates new geometry element using elementReaders,
     * but it also support references and can return existing elements.
     * @param source xml data source from which element data should be read
     * @return element which was read and create or to which reference was read
     * @throw GeometryElementNamesConflictException if element with read name already exists
     * @throw NoSuchGeometryElement if ref element reference to element which not exists
     * @throw NoAttrException if XML tag has no required attribiutes
     */
    GeometryElement& readElement(XMLReader& source);
    
    /**
     * Call readElement(source) and try dynamic cast it to @a RequiredElementType.
     * @param source xml data source from which element data should be read
     * @return element (casted to RequiredElementType) which was read and create or to which reference was read
     * @tparam RequiredElementType required type of element
     * @throw UnexpectedGeometryElementTypeException if requested element is not of type RequiredElementType
     * @throw GeometryElementNamesConflictException if element with read name already exists
     * @throw NoSuchGeometryElement if ref element reference to element which not exists
     * @throw NoAttrException if XML tag has no required attribiutes
     */
    template <typename RequiredElementType>
    RequiredElementType& readElement(XMLReader& source);
    
    /**
     * Read all elements up to end of XML tag and call functor(element) for each element which was read.
     * @param source
     * @param functor
     * @tparam FunctorType unary functor which can take RequiredElementType& as argument
     * @tparam RequiredElementType required type of element
     */
    template <typename FunctorType, typename RequiredElementType = GeometryElement>
    void readAllElements(XMLReader& source, FunctorType functor);
};

//specialization for most types
template <typename RequiredElementType>
inline RequiredElementType& GeometryManager::readElement(XMLReader& source) {
    RequiredElementType* result = dynamic_cast<RequiredElementType*>(&readElement(source));
    if (!result) throw UnexpectedGeometryElementTypeException();
    return *result;
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline GeometryElement& GeometryManager::readElement<GeometryElement>(XMLReader& source) {
    return readElement(source);
}

template <typename FunctorType, typename RequiredElementType>
inline void GeometryManager::readAllElements(XMLReader& source, FunctorType functor) {
    while(source.read()) {
        switch (source.getNodeType()) {
            case irr::io::EXN_ELEMENT_END: return;  
            case irr::io::EXN_ELEMENT: functor(readElement<RequiredElementType>(source));
            //TODO what with all other XML types (which now are just ignored)?
        }
    }
}

}	// namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
