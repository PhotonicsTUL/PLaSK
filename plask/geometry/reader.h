#ifndef PLASK__GEOMETRY_READER_H
#define PLASK__GEOMETRY_READER_H

#include "../utils/xml.h"
#include "../axes.h"
#include "manager.h"
#include "../material/db.h"

namespace plask {

/// Value for expected suffix for names of 2d elements types, see GeometryReader::expectedSuffix.
#define PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D "2d"

/// Value for expected suffix for names of 3d elements types, see GeometryReader::expectedSuffix.
#define PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D "3d"

/**
 * Allow to read geometry from XML.
 *
 * Have references to both: XML data source reader and geometry manager.
 * Manage names of axis while reading.
 */
struct GeometryReader {

    /**
     * Create new geometry element with parameters reading from XML source.
     *
     * After return reader should point to end of tag of this element.
     * Can call managers methods to read children (GeometryReader::readElement).
     * Should throw exception if can't create element.
     */
    typedef shared_ptr<GeometryElement> element_read_f(GeometryReader& reader);

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
     * Each element can create one global instance of this class to register own reader.
     */
    struct RegisterElementReader {
        RegisterElementReader(const std::string& tag_name, element_read_f* reader) {
            GeometryReader::registerElementReader(tag_name, reader);
        }
    };

    /// Current names of axis.
    const AxisNames* axisNames;

    /**
     * Currently expected suffix for names of geometry elements types, can have one of the following values:
     * - 0 dimensions of children space can't be deduced (initial value),
     * - PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D if 2d children are expected,
     * - PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D if 3d children are expected.
     */
    const char* expectedSuffix;

    /// Material database used by geometry (leafs).
    const MaterialsDB& materialsDB;

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
    std::string getAxisLonName() { return getAxisName(axis::lon_index); }

    /**
     * Get current tran direction axis name.
     * @return name of tran axis
     */
    std::string getAxisTranName() { return getAxisName(axis::tran_index); }

    /**
     * Get current up direction axis name.
     * @return name of up axis
     */
    std::string getAxisUpName() { return getAxisName(axis::up_index); }

    /**
     * Read axis name from current reader tag, set it in reader as current,
     * and restore old axisNames value when out of the scope.
     */
    struct ReadAxisNames {
        GeometryReader& reader;
        const AxisNames* old;
        ReadAxisNames(GeometryReader& reader);
        ~ReadAxisNames() { reader.axisNames = old; }
    };

    /**
     * Store current expectedSuffix, set new one, and restore old when out of the scope (in destructor).
     */
    struct SetExpectedSuffix {
        GeometryReader& reader;
        const char* old;
        SetExpectedSuffix(GeometryReader& reader, const char* new_expected_suffix);
        ~SetExpectedSuffix() { reader.expectedSuffix = old; }
    };

    /// Geometry manager which store reading results.
    GeometryManager& manager;

    /// XML data source
    XMLReader& source;

    /**
     * @param manager
     * @param source xml data source from which element data should be read
     * @param materialsDB materials database used to set leafs materials
     */
    GeometryReader(GeometryManager& manager, XMLReader& source, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Read geometry element from @p source and add it GeometryManager structures.
     *
     * Typically it creates new geometry element using elementReaders,
     * but it also support references and can return existing elements.
     *
     * After call source reader point to end of tag which represent read element.
     * @return element which was read and create or to which reference was read
     * @throw GeometryElementNamesConflictException if element with read name already exists
     * @throw NoSuchGeometryElement if ref element reference to element which not exists
     * @throw NoAttrException if XML tag has no required attributes
     */
    shared_ptr<GeometryElement> readElement();

    /**
     * Helper function to read elements which have exactly one child (typically: transform).
     *
     * Befor call source reader should point to parent element tag (typically transform element)
     * and after call it will be point to end of parent element tag.
     * @return child element which was read and create or to which reference was read
     */
    shared_ptr<GeometryElement> readExactlyOneChild();

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
    shared_ptr<RequiredElementType> readElement();

    /**
     * Call readExactlyOneChild() and try dynamic cast it to @a RequiredElementType.
     * @return element (casted to RequiredElementType) which was return by readExactlyOneChild()
     * @tparam RequiredElementType required type of element
     */
    template <typename RequiredElementType>
    shared_ptr<RequiredElementType> readExactlyOneChild();

    /**
     * Try reading calculation space. Throw exception if can't.
     * @return calculation space which was read
     */
    shared_ptr<CalculationSpace> readCalculationSpace();

};

// specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryReader::readElement() {
    shared_ptr<RequiredElementType> result = dynamic_pointer_cast<RequiredElementType>(readElement());
    if (!result) throw UnexpectedGeometryElementTypeException();
    return result;
}

// specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryElement> GeometryReader::readElement<GeometryElement>() {
    return readElement();
}

// specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryReader::readExactlyOneChild() {
    shared_ptr<RequiredElementType> result = dynamic_pointer_cast<RequiredElementType>(readExactlyOneChild());
    if (!result) throw UnexpectedGeometryElementTypeException();
    return result;
}

// specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryElement> GeometryReader::readExactlyOneChild<GeometryElement>() {
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
