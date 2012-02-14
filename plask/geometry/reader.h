#ifndef PLASK__GEOMETRY_READER_H
#define PLASK__GEOMETRY_READER_H

#include "../utils/xml.h"
#include "manager.h"

namespace plask {

//struct GeometryManager;

/**
 * Hold names of axises.
 * 
 * Can change: axis number (from 0 to 2) <-> axis name (string)
 */
struct AxisNames {
    
    /**
     * Register of axis names.
     */
    struct Register {
        ///Name of system of axis names -> AxisNames
        std::map<std::string, AxisNames> axisNames;
        
        ///Construct empty register.
        Register() {}
        
        template<typename... Params>
        Register(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const Params&... names) {
            this->operator()(c0_name, c1_name, c2_name, names...);
        }
        
        /**
         * Add axis names to register.
         * @param c0_name, c1_name, c2_name axis names
         * @param name name of axis names, register key
         */
        void addname(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const std::string& name) {
            axisNames[name] = AxisNames(c0_name, c1_name, c2_name);
        }
        
        /**
         * Add axis names using as key: c0_name + c1_name + c2_name
         * @param c0_name, c1_name, c2_name axis names
         */
        Register& operator()(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name) {
            addname(c0_name, c1_name, c2_name, c0_name + c1_name + c2_name);
            return *this;
        }
        
        /**
         * Add axis names to register using as keys given @p name and c0_name + c1_name + c2_name.
         * @param c0_name, c1_name, c2_name axis names
         * @param name name of axis names, register key
         * @tparam Param1 std::string or const char*
         */   
        template<typename Param1>
        Register& operator()(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const Param1& name) {
            addname(c0_name, c1_name, c2_name, name);
            return this->operator()(c0_name, c1_name, c2_name);
        }

        /**
         * Add axis names to register using as keys given names and c0_name + c1_name + c2_name.
         * @param c0_name, c1_name, c2_name axis names
         * @param firstName, names names of axis names, register keys
         * @tparam Param1, Params each of type std::string or const char*
         */
        template<typename Param1, typename... Params>
        Register& operator()(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const Param1& firstName, const Params&... names) {
            addname(c0_name, c1_name, c2_name, firstName);
            return this->operator()(c0_name, c1_name, c2_name, names...);
        }
        
        /**
         * Get axis names with given name (key).
         * @param name register keys
         * @return axis names
         * @throw NoSuchAxisNames if axis names with given name not exists in register
         */
        const AxisNames& get(const std::string& name) const;
    };
    
    ///Name of axises (by index).
    std::string byIndex[3];
    
    ///Construct uninitialized object. Do nothing.
    AxisNames() {}
    
    AxisNames(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name);
    
    /**
     * Get axis name by index.
     * @param i index of axis name, from 0 to 2
     * @return name of i-th axis
     */
    const std::string& operator[](const std::size_t i) const { return byIndex[i]; }
    
    /**
     * Get axis index by name.
     * @param name axis name
     * @return index (from 0 to 2) of axis with given @p name or 3 if no axis with given name
     */
    std::size_t operator[](const std::string& name) const;
    
};

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
    
    ///Register of standard axis names.
    static AxisNames::Register axisNamesRegister;
    
    ///Current names of axis.
    const AxisNames* axisNames;
    
    ///Material database used by geometry (leafs).
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

    ///Geometry manager which store reading results.
    GeometryManager& manager;

    ///XML data source
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
};

//specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryReader::readElement() {
    shared_ptr<RequiredElementType> result = dynamic_pointer_cast<RequiredElementType>(readElement());
    if (!result) throw UnexpectedGeometryElementTypeException();
    return result;
}

//specialization for GeometryElement which doesn't required dynamic_cast
template <>
inline shared_ptr<GeometryElement> GeometryReader::readElement<GeometryElement>() {
    return readElement();
}

//specialization for most types
template <typename RequiredElementType>
inline shared_ptr<RequiredElementType> GeometryReader::readExactlyOneChild() {
    shared_ptr<RequiredElementType> result = dynamic_pointer_cast<RequiredElementType>(readExactlyOneChild());
    if (!result) throw UnexpectedGeometryElementTypeException();
    return result;
}

//specialization for GeometryElement which doesn't required dynamic_cast
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
