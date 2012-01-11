#ifndef PLASK__EXCEPTIONS_H
#define PLASK__EXCEPTIONS_H

/** @file
This file includes definitions of all exceptions classes which are used in PLaSK.
*/

#include <stdexcept>
#include "utils/format.h"

namespace plask {

/**
 * Base class for all exceptions thrown by plask library.
 */
struct Exception: public std::runtime_error {
    
    ///@param msg error message
    Exception(const std::string& msg): std::runtime_error(msg) {}
    
    /**
     * Format error message using boost::format.
     */
    template <typename... T>
    Exception(const std::string& msg, const T&... args): std::runtime_error(format(msg, args...)) {
    }
};

/**
 * Exceptions of this class are throw in cases of critical and very unexpected errors (possible plask bugs).
 */
struct CriticalException: public Exception {
    
    ///@param msg error message
    CriticalException(const std::string& msg): Exception(msg) {}
};

/**
 * This exception is thrown when some method is not implemented.
 */
struct NotImplemented: public Exception {
    //std::string methodName;

    ///@param method_name name of not implemented method
    NotImplemented(const std::string& method_name)
    : Exception("Method not implemented: " + method_name)/*, methodName(method_name)*/ {}

    /**
     * @param where method is not implemented (typically class name)
     * @param method_name name of not implemented method
     */
    NotImplemented(const std::string& where, const std::string& method_name)
    : Exception("In " + where + ": Method not implemented: " + method_name)/*, methodName(method_name)*/ {}
};

/**
 * This exception is thrown when some value (function argument) out of bound.
 */
struct OutOfBoundException: public Exception {
    
    ///@param msg error message
    OutOfBoundException(const std::string& where, const std::string& argname)
        : Exception("%1%: argument %2% out of bound", where, argname) {}
    
    template <typename BoundType>
    OutOfBoundException(const std::string& where, const std::string& argname, const BoundType& was, const BoundType& lo, const BoundType& hi)
        : Exception("%1%: argument %2% out of bound, should be between %3% and %4%, but was %5%.", where, argname, lo, hi, was) {}
};

//-------------- Connected with providers/receivers: -----------------------

/**
 * This exception is thrown, typically on access to @ref plask::Receiver "receiver" data,
 * when there is no @ref plask::Provider "provider" connected with it.
 * @see @ref providers
 */
struct NoProvider: public Exception {
    NoProvider(): Exception("No provider.") {}
};



/*
 * Exceptions of this class are throw when some string parser find errors.
 */
/*struct ParseException: public Exception {
    ParseException(): Exception("Parse error.") {}
    ParseException(std::string& msg): Exception("Parse error: " + msg) {}
};*/

//-------------- Connected with materials: -----------------------

/**
 * This exception is thrown when material (typically with given name) is not found.
 */
struct NoSuchMaterial: public Exception {
    //std::string materialName;

    ///@param material_name name of material which not exists
    NoSuchMaterial(const std::string& material_name)
        : Exception("No such material \"" + material_name + "\"")/*, materialName(material_name)*/ {}
};

/**
 * This exception is thrown by material methods which are not implemented.
 */
struct MaterialMethodNotImplemented: public NotImplemented {

    /**
     * @param material_name name of material
     * @param method_name name of not implemented method
     */
    MaterialMethodNotImplemented(const std::string& material_name, const std::string& method_name)
    : NotImplemented("material " + material_name, method_name) {
    }

};

/**
 * Exceptions of this class are throw when material string parser find errors.
 */
struct MaterialParseException: public Exception {
    MaterialParseException(): Exception("Material parse error.") {}
    ///@param msg error message
    MaterialParseException(const std::string& msg): Exception("Material parse error: " + msg) {}

    template <typename... T>
    MaterialParseException(const std::string& msg, const T&... args): Exception("Material parse error: " + msg, args...) {
    }
};

//-------------- Connected with XML: -----------------------
/**
 * Exceptions of this class are throw when required attribute is not found in XML tag.
 */
struct XMLNoAttrException: public Exception {
    /**
     * @param where where (typically in which tag) there are no required attribiute
     * @param attr_name name of required attribiute
     */
    XMLNoAttrException(const std::string& where, const std::string& attr_name): Exception(where + ": XML tag has no required attribute \"" + attr_name + "\"") {}
};

/**
 * Exceptions of this class are throw when XML file/data stream unexpected end.
 */
struct XMLUnexpectedEndException: public Exception {
    XMLUnexpectedEndException(): Exception("Unexpected end of XML data.") {}
};

/**
 * Exceptions of this class are throw when type of XML element is other than expectation.
 */
struct XMLUnexpectedElementException: public Exception {
    /**
     * @param what_is_expected what is expected (typically tag with name, etc.)
     */
    XMLUnexpectedElementException(const std::string& what_is_expected): Exception("There is no expected " + what_is_expected + " in XML.") {}
};

//-------------- Connected with geometry: -----------------------

/**
 * Exceptions of this class are throw by some geometry element classes when there is no required child.
 */
struct NoChildException: public Exception {
    NoChildException(): Exception("No child.") {}
};

/**
 * This exception is thrown when geometry element (typically with given name) is not found.
 */
struct NoSuchGeometryElementType: public Exception {
    //std::string materialName;

    /**
     * @param element_type_name name of element type which is not found
     */
    NoSuchGeometryElementType(const std::string& element_type_name)
        : Exception("No geometry element with given type name \"" + element_type_name + "\"")/*, materialName(material_name)*/ {}
};

/**
 * Exceptions of this class are throw by some geometry element classes when there is no required child.
 */
struct GeometryElementNamesConflictException: public Exception {
    
    /**
     * @param element_name name of element which is already exists
     */
    GeometryElementNamesConflictException(const std::string& element_name): Exception("Geometry element with given name \"" + element_name + "\" already exists.") {}
};

/**
 * This exception is thrown when geometry element (typically with given name) is not found.
 */
struct NoSuchGeometryElement: public Exception {
    //std::string materialName;

    /**
     * @param element_name name of element which is not found
     */
    NoSuchGeometryElement(const std::string& element_name)
    : Exception("No geometry element with name " + element_name)/*, materialName(material_name)*/ {}
};

/**
 * This exception is thrown when geometry element has type different than expectation (for example is 3d but expected 2d).
 */
struct UnexpectedGeometryElementTypeException: public Exception {
    UnexpectedGeometryElementTypeException(): Exception("Geometry element has unexpected type.") {}
};

/**
 * This exception is thrown when axis (typically with given name) is not found.
 */
struct NoSuchAxisNames: public Exception {
    //std::string materialName;

    ///@param axis_names name of axis which not exists
    NoSuchAxisNames(const std::string& axis_names)
        : Exception("No such axis names \"%1%\".", axis_names) {}
};


} // namespace plask

#endif  //PLASK__EXCEPTIONS_H
