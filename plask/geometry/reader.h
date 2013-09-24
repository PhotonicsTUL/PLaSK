#ifndef PLASK__GEOMETRY_READER_H
#define PLASK__GEOMETRY_READER_H

#include "../utils/xml/reader.h"
#include "../axes.h"
#include "../material/db.h"
#include "space.h"
#include <functional>
#include "../mesh/boundary_conditions.h"

namespace plask {

struct Manager;

/**
 * Allow to read geometry from XML.
 *
 * Have references to both: XML data source reader and geometry manager.
 * Manage names of axis while reading.
 */
class GeometryReader {

    /// Allow to access objects by auto-name (with names beggined with '#').
    std::map<std::string, shared_ptr<GeometryObject> > autoNamedObjects;

  public:

    static constexpr const char* XML_NAME_ATTR = "name";                        ///< name of object's/geometry's name attribute in XML
    static constexpr const char* XML_MATERIAL_ATTR = "material";                ///< name of material attribute in XML
    static constexpr const char* XML_MATERIAL_TOP_ATTR = "topmaterial";          ///< name of top material attribute in XML
    static constexpr const char* XML_MATERIAL_BOTTOM_ATTR = "bottommaterial";    ///< name of bottom material attribute in XML

    /**
     * Create new geometry object with parameters reading from XML source.
     *
     * After return reader should point to end of tag of this object.
     * Can call managers methods to read children (GeometryReader::readObject).
     * Should throw exception if can't create object.
     */
    typedef shared_ptr<GeometryObject> object_read_f(GeometryReader& reader);

    /**
     * @return Global objects readers register.
     * Map: xml tag name -> object reader function.
     */
    static std::map<std::string, object_read_f*>& objectReaders();

    /**
     * Add reader to objectReaders.
     * @param tag_name XML tag name
     * @param reader object reader function
     */
    static void registerObjectReader(const std::string& tag_name, object_read_f* reader);

    /**
     * Helper which call registerObjectReader in constructor.
     *
     * Each object can create one global instance of this class to register own reader.
     */
    struct RegisterObjectReader {
        RegisterObjectReader(const std::string& tag_name, object_read_f* reader) {
            GeometryReader::registerObjectReader(tag_name, reader);
        }
    };

    /**
     * Currently expected suffix for names of geometry objects types, can have one of the following values:
     * - 0 dimensions of children space can't be deduced (initial value),
     * - PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D if 2d children are expected,
     * - PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D if 3d children are expected.
     */
    const char* expectedSuffix;

    const AxisNames& getAxisNames() const;

    /**
     * Get current axis name.
     * @param axis_index axis index
     * @return name of axis which have an @p axis_index
     */
    std::string getAxisName(std::size_t axis_index);

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
     * Store current expectedSuffix, set new one, and restore old when out of the scope (in destructor).
     */
    struct SetExpectedSuffix {
        GeometryReader& reader;
        const char* old;
        SetExpectedSuffix(GeometryReader& reader, const char* new_expected_suffix);
        ~SetExpectedSuffix() { reader.expectedSuffix = old; }
    };

    /// Geometry manager which stores reading results.
    Manager& manager;

    /// XML data source
    XMLReader& source;

    /// Source of materials, typically use material database.
    MaterialsSource materialSource;

    /**
     * Get material from material source (typically material database) connected with this reader.
     *
     * Throw excpetion if can't get material (no material with given name, etc.).
     * @param material_full_name full material name to get
     * @return material with name @p material_full_name
     */
    shared_ptr<Material> getMaterial(const std::string& material_full_name) const {
        return materialSource(material_full_name);
    }

    /**
     * Read material from XML source (from attribute with name XML_MATERIAL_ATTR).
     *
     * Throw exception if there is no XML_MATERIAL_ATTR attribute or can't get material (no material with given name, etc.).
     * @return material which was read
     */
    shared_ptr<Material> requireMaterial() const {
        return getMaterial(source.requireAttribute(XML_MATERIAL_ATTR));
    }

    /**
     * @param manager
     * @param source xml data source from which object data should be read
     * @param materialsDB materials database used to set leafs materials
     */
    GeometryReader(Manager& manager, XMLReader& source, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * @param manager
     * @param source xml data source from which object data should be read
     * @param materialsSource materials source used to set leafs materials
     */
    GeometryReader(Manager& manager, XMLReader& source, const MaterialsSource& materialsSource);

    /**
     * Read geometry object from @p source and add it Manager structures.
     *
     * Typically it creates new geometry object using objectReaders,
     * but it also support references and can return existing objects.
     *
     * After call source reader point to end of tag which represent read object.
     * @return object which was read and create or to which reference was read
     * @throw NamesConflictException if object with read name already exists
     * @throw NoSuchGeometryObject if 'again' object references to object which does not exists
     * @throw NoAttrException if XML tag has no required attributes
     */
    shared_ptr<GeometryObject> readObject();

    /**
     * Helper function to read objects which have exactly one child (typically: transform).
     *
     * Befor call source reader should point to parent object tag (typically transform object)
     * and after call it will be point to end of parent object tag.
     * @return child object which was read and create or to which reference was read
     */
    shared_ptr<GeometryObject> readExactlyOneChild();

    /**
     * Call readObject() and try dynamic cast it to @a RequiredObjectType.
     * @return object (casted to RequiredObjectType) which was read and create or to which reference was read
     * @tparam RequiredObjectType required type of object
     * @throw UnexpectedGeometryObjectTypeException if requested object is not of type RequiredObjectType
     * @throw NamesConflictException if object with read name already exists
     * @throw NoSuchGeometryObject if 'again' object references to object which does not exists
     * @throw NoAttrException if XML tag has no required attributes
     */
    template <typename RequiredObjectType>
    shared_ptr<RequiredObjectType> readObject();

    /**
     * Call readExactlyOneChild() and try dynamic cast it to @a RequiredObjectType.
     * @return object (casted to RequiredObjectType) which was return by readExactlyOneChild()
     * @tparam RequiredObjectType required type of object
     */
    template <typename RequiredObjectType>
    shared_ptr<RequiredObjectType> readExactlyOneChild();

    /**
     * Try reading calculation space. Throw exception if can't.
     * @return calculation space which was read
     */
    shared_ptr<Geometry> readGeometry();

    /**
     * Get named objects. It support boths: named objects (from manager) and auto-named objects.
     * @param name object name (can be auto-generated: in form '#'+number)
     * @return object with given name
     * @throw NoSuchGeometryObject if object was not found
     */
    shared_ptr<GeometryObject> requireObjectWithName(const std::string& name) const;

    /**
     * Add name of object to register.
     *
     * It throws excepetion in case of names conflict.
     * @param name name of given @p object (can be auto-generated: in form '#'+number)
     * @param object geometry object which should be available by given @p name
     */
    void registerObjectName(const std::string& name, shared_ptr<GeometryObject> object);

    /**
     * Add name of object to register. Do nothing if @p name has no value.
     *
     * It throws excepetion in case of names conflict.
     * @param name name of given @p object (can be auto-generated: in form '#'+number)
     * @param object geometry object which should be available by given @p name
     */
    void registerObjectName(const boost::optional<std::string>& name, shared_ptr<GeometryObject> object) {
        if (name) registerObjectName(*name, object);
    }

    /**
     * Add name of object to register. Do nothing if @p name has no value.
     *
     * It throws excepetion in case of names conflict.
     * @param name name of given @p object (can be auto-generated: in form '#'+number)
     * @param object geometry object which should be available by given @p name
     */
    void registerObjectName(const boost::optional<const std::string>& name, shared_ptr<GeometryObject> object) {
        if (name) registerObjectName(*name, object);
    }


    /**
     * Add name of object to register. Read name from current, XML source tag.
     * Do nothing if name attribute is not available.
     *
     * It throws excepetion in case of names conflict.
     * @param object geometry object which should be available by name which was read from current tag
     */
    void registerObjectNameFromCurrentNode(shared_ptr<GeometryObject> object) {
        registerObjectName(source.getAttribute(XML_NAME_ATTR), object);
    }

};

// specialization for most types
template <typename RequiredObjectType>
inline shared_ptr<RequiredObjectType> GeometryReader::readObject() {
    shared_ptr<RequiredObjectType> result = dynamic_pointer_cast<RequiredObjectType>(readObject());
    if (!result) throw UnexpectedGeometryObjectTypeException();
    return result;
}

// specialization for GeometryObject which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryObject> GeometryReader::readObject<GeometryObject>() {
    return readObject();
}

// specialization for most types
template <typename RequiredObjectType>
inline shared_ptr<RequiredObjectType> GeometryReader::readExactlyOneChild() {
    shared_ptr<RequiredObjectType> result = dynamic_pointer_cast<RequiredObjectType>(readExactlyOneChild());
    if (!result) throw UnexpectedGeometryObjectTypeException();
    return result;
}

// specialization for GeometryObject which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryObject> GeometryReader::readExactlyOneChild<GeometryObject>() {
    return readExactlyOneChild();
}

/*template <typename FunctorType, typename RequiredObjectType>
inline void GeometryReader::readAllObjects(XMLReader& source, FunctorType functor) {
    while(source.read()) {
        switch (source.getNodeType()) {
            case irr::io::EXN_ELEMENT_END: return;
            case irr::io::EXN_ELEMENT: functor(readObject<RequiredObjectType>(source));
            //TODO what with all other XML types (which now are just ignored)?
        }
    }
}*/

}   // namespace plask

#endif // PLASK__GEOMETRY_READER_H
