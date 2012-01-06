#ifndef PLASK__GEOMETRY_READER_H
#define PLASK__GEOMETRY_READER_H

#include "../utils/xml.h"
#include "manager.h"

namespace plask {

//struct GeometryManager;

struct GeometryReader {

    /**
     * Create new geometry element (using new operator) with parameters reading from XML source.
     * Can call managers methods to read children (GeometryManager::readElement).
     * Should throw exception if can't create element.
     * Result will be delete (using delete operator) by caller.
     */
    typedef GeometryElement* element_read_f(GeometryReader& reader);

    /**
     * @return Global elements readers register.
     * Map: xml tag name -> element reader function.
     */
    static std::map<std::string, element_read_f*>& elementReaders();

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
        RegisterElementReader(const std::string& tag_name, element_read_f* reader) {
            GeometryReader::registerElementReader(tag_name, reader);
        }
    };

    GeometryManager& manager;

    ///XML data source
    XMLReader& source;

    /**
     * @param manager
     * @param source xml data source from which element data should be read
     */
    GeometryReader(GeometryManager& manager, XMLReader& source);

    /**
     * Read geometry element from @a source and add it GeometryManager structures.
     *
     * Typically it creates new geometry element using elementReaders,
     * but it also support references and can return existing elements.
     * @return element which was read and create or to which reference was read
     * @throw GeometryElementNamesConflictException if element with read name already exists
     * @throw NoSuchGeometryElement if ref element reference to element which not exists
     * @throw NoAttrException if XML tag has no required attributes
     */
    GeometryElement& readElement();

    /**
     * Skip current element in source and read exactly one geometry element (which also skip).
     */
    GeometryElement& readExactlyOneChild();

    /**
     * Call readElement() and try dynamic cast it to @a RequiredElementType.
     * @return element (casted to RequiredElementType) which was read and create or to which reference was read
     * @tparam RequiredElementType required type of element
     * @throw UnexpectedGeometryElementTypeException if requested element is not of type RequiredElementType
     * @throw GeometryElementNamesConflictException if element with read name already exists
     * @throw NoSuchGeometryElement if ref element reference to element which not exists
     * @throw NoAttrException if XML tag has no required attributes
     */
    template <typename RequiredElementType>
    RequiredElementType& readElement();

    template <typename RequiredElementType>
    RequiredElementType& readExactlyOneChild();
};

//specialization for most types
template <typename RequiredElementType>
inline RequiredElementType& GeometryReader::readElement() {
    RequiredElementType* result = dynamic_cast<RequiredElementType*>(&readElement());
    if (!result) throw UnexpectedGeometryElementTypeException();
    return *result;
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline GeometryElement& GeometryReader::readElement<GeometryElement>() {
    return readElement();
}

//specialization for most types
template <typename RequiredElementType>
inline RequiredElementType& GeometryReader::readExactlyOneChild() {
    RequiredElementType* result = dynamic_cast<RequiredElementType*>(&readExactlyOneChild());
    if (!result) throw UnexpectedGeometryElementTypeException();
    return *result;
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline GeometryElement& GeometryReader::readExactlyOneChild<GeometryElement>() {
    return readExactlyOneChild();
}

/*template <typename FunctorType, typename RequiredElementType>
inline void GeometryReader::readAllElements(XMLReader& source, FunctorType functor) {
    while(source.read()) {
        switch (source.getNodeType()) {
            case irr::io::EXN_ELEMENT_END: return;
            case irr::io::EXN_ELEMENT: functor(readElement<RequiredElementType>(source));
            //TODO what with all other XML types (which now are just ignored)?
        }
    }
}*/

}   // namespace plask

#endif // PLASK__GEOMETRY_READER_H
