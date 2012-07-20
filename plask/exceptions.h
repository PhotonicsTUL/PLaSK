#ifndef PLASK__EXCEPTIONS_H
#define PLASK__EXCEPTIONS_H

/** @file
This file includes definitions of all exceptions classes which are used in PLaSK.
*/

#include <stdexcept>
#include "utils/format.h"
#include "utils/string.h"

namespace plask {

/**
 * Base class for all exceptions thrown by plask library.
 */
struct Exception: public std::runtime_error {

    /// @param msg error message
    Exception(const std::string& msg): std::runtime_error(msg) {}

    /**
     * Format error message using boost::format.
     */
    template <typename... T>
    Exception(const std::string& msg, const T&... args): std::runtime_error(format(msg, args...)) {
    }
};

/**
 * Exceptions of this class are thrownin cases of critical and very unexpected errors (possible plask bugs).
 */
struct CriticalException: public Exception {

    /// @param msg error message
    CriticalException(const std::string& msg): Exception("Critical exception: " + msg) {}

    template <typename... T>
    CriticalException(const std::string& msg, const T&... args): Exception("Critical exception: " + msg, args...) {}
};

/**
 * This exception is thrown when some method is not implemented.
 */
struct NotImplemented: public Exception {
    //std::string methodName;

    /// @param method_name name of not implemented method
    NotImplemented(const std::string& method_name)
    : Exception("Method not implemented: " + method_name)/*, methodName(method_name)*/ {}

    /**
     * @param where method is not implemented (typically class name)
     * @param method_name name of not implemented method
     */
    NotImplemented(const std::string& where, const std::string& method_name)
    : Exception(where + ": Method not implemented: " + method_name)/*, methodName(method_name)*/ {}
};

/**
 * This exception is thrown when some value (function argument) is out of bound.
 */
struct OutOfBoundException: public Exception {

    OutOfBoundException(const std::string& where, const std::string& argname)
        : Exception("%1%: argument %2% out of bound", where, argname) {}

    template <typename BoundTypeWas, typename BoundTypeLo, typename BoundTypeHi>
    OutOfBoundException(const std::string& where, const std::string& argname, const BoundTypeWas& was, const BoundTypeLo& lo, const BoundTypeHi& hi)
        : Exception("%1%: argument %2% out of bound, should be between %3% and %4%, but was %5%", where, argname, lo, hi, was) {}
};

/**
 * This exception is thrown if there is a problem with dimensions.
 */
struct DimensionError: public Exception {
    template <typename... T>
    DimensionError(T... args) : Exception(args...) {}
};

/**
 * This exception is thrown when value specified by the user is bad
 */
struct BadInput: public Exception {

    /**
     * @param where name of class/function/operation doing the computations
     * @param msg error message (format)
     * @param params formating parmeters for msg
     */
    template <typename... Params>
    BadInput(const std::string& where, const std::string& msg, Params... params)
        : Exception("%1%: %2%", where, format(msg, params...)) {};
};

/**
 * This exception shoulb be thrown by modules in case of error in computations.
 */
struct ComputationError: public Exception {

    /**
     * @param where name of class/function/operation doing the computations
     * @param msg error message
     */
    ComputationError(const std::string& where, const std::string& msg)
        : Exception("%1%: %2%", where, msg) {};
};

/**
 * This is throwed if name is bad id.
 */
struct BadId: public Exception {

    BadId(const std::string& where, const char* str_to_check, char underline_ch = '_')
        : Exception("\"%1%\" is bad name for %2%, this name shouldn't be empty and should consists of english letters, '%3%' character and digits (but not at beggining).", str_to_check, where, underline_ch) {};

    static void throwIfBad(const std::string& where, const char* str_to_check, char underline_ch = '_') {
        if (!isCid(str_to_check, underline_ch))
            throw BadId(where, str_to_check, underline_ch);
    }

    static void throwIfBad(const std::string& where, const std::string& str_to_check, char underline_ch = '_') {
        throwIfBad(where, str_to_check.c_str(), underline_ch);
    }

};

//-------------- Connected with providers/receivers: -----------------------

/**
 * This exception is thrown, typically on access to @ref plask::Receiver "receiver" data,
 * when there is no @ref plask::Provider "provider" connected with it.
 * @see @ref providers
 */
struct NoProvider: public Exception {
    NoProvider(): Exception("No provider") {}
    NoProvider(const char* provider_name): Exception("No %1% set nor its provider connected", provider_name) {}
};

struct NoValue: public Exception {
    NoValue(): Exception("No value") {}
    NoValue(const char* provider_name): Exception("%1% cannot be provided now", [](std::string s)->std::string{s[0]=std::toupper(s[0]);return s;}(provider_name) ) {}
};


/*
 * Exceptions of this class are thrownwhen some string parser find errors.
 */
/*struct ParseException: public Exception {
    ParseException(): Exception("Parse error") {}
    ParseException(std::string& msg): Exception("Parse error: " + msg) {}
};*/

//-------------- Connected with materials: -----------------------

/**
 * This exception is thrown when material (typically with given name) is not found.
 */
class NoSuchMaterial: public Exception {

    template <typename ComponentMap>
    std::string constructMsg(const ComponentMap& comp, const std::string dopant_name) {
        std::string result = "No material with composition consisting of:";
        for (auto c: comp) (result += ' ') += c.first;
        if (!dopant_name.empty()) (result += ", doped with: ") += dopant_name;
        return result;
    }

public:
    /// @param material_name name of material which not exists
    NoSuchMaterial(const std::string& material_name)
        : Exception("No such material: %1%", material_name)/*, materialName(material_name)*/ {}

    NoSuchMaterial(const std::string& material_name, const std::string& dopant_name)
        : Exception("No such material: %1%%2%%3%", material_name, dopant_name.empty() ? "" : ":", dopant_name)/*, materialName(material_name)*/ {}

    template <typename ComponentMap>
    NoSuchMaterial(const ComponentMap& comp, const std::string dopant_name)
        : Exception(constructMsg(comp, dopant_name)) {}

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
    : NotImplemented("Material " + material_name, method_name) {
    }

};

/**
 * This exception is thrown by if some material property does not make sense for particular material.
 */
struct MaterialMethodNotApplicable: public Exception {

    /**
     * @param material_name name of material
     * @param method_name name of not implemented method
     */
    MaterialMethodNotApplicable(const std::string& material_name, const std::string& method_name)
    : Exception("Material %1%: method not applicable: %2%", material_name, method_name) {
    }

};

/**
 * Exceptions of this class are thrownwhen material string parser find errors.
 */
struct MaterialParseException: public Exception {
    MaterialParseException(): Exception("Material parse error") {}
    /// @param msg error message
    MaterialParseException(const std::string& msg): Exception("Material parse error: " + msg) {}

    template <typename... T>
    MaterialParseException(const std::string& msg, const T&... args): Exception("Material parse error: " + msg, args...) {
    }
};

class MaterialCantBeMixedException: public Exception {

    /*template <typename ComponentMap>
    std::string constructMsg(const ComponentMap& comp, const std::string dopant_name) {
        std::string result = "material with composition consisting of:";
        for (auto c: comp) (result += ' ') += c.first;
        if (!dopant_name.empty()) (result += ", doped with: ") += dopant_name;
        return result;
    }*/

public:
    MaterialCantBeMixedException(const std::string& material1name_with_components, const std::string& material2name_with_components, const std::string& common_dopant = "")
        : Exception("Material \"%1%%3%\" can not be mixed with material \"%2%%3%\"",
              material1name_with_components, material2name_with_components, common_dopant.empty() ? "" : ':' + common_dopant)
              {}

};

//-------------- Connected with XML: -----------------------
/**
 * Exceptions of this class are thrown when the required attribute is not found in XML tag.
 */
struct XMLNoAttrException: public Exception {
    /**
     * @param where where (typically in which tag) there are no required attribiute
     * @param attr_name name of required attribiute
     */
    XMLNoAttrException(const std::string& where, const std::string& attr_name): Exception(where + ": XML tag has no required attribute \"" + attr_name + "\"") {}
};

/**
 * Exceptions of this class are thrown when the attribute has wrong value.
 */
struct XMLBadAttrException: public Exception {
    /**
     * @param where where (typically in which tag) there is bad value for atribiute
     * @param attr_name name of attribiute
     * @param attr_value illegal value of attribiute
     */
    XMLBadAttrException(const std::string& where, const std::string& attr_name, const std::string& attr_value):
        Exception(where + ": XML tag attribute \"" + attr_name + "\" has bad value \"" + attr_value + "\"") {}
};

/**
 * Exceptions of this class are thrownwhen XML file/data stream unexpected end.
 */
struct XMLUnexpectedEndException: public Exception {
    XMLUnexpectedEndException(): Exception("Unexpected end of XML data") {}
};

/**
 * Exceptions of this class are thrownwhen the type of XML element is different than expected.
 */
struct XMLUnexpectedElementException: public Exception {
    /**
     * @param what_is_expected what is expected (typically tag with name, etc.)
     */
    XMLUnexpectedElementException(const std::string& what_is_expected): Exception("Expected " + what_is_expected + " in XML") {}
};

/**
 * Exceptions of this class are thrown when the value of an XML attribute is different than expected.
 */
struct XMLUnexpectedAttributeValueException: public Exception {
    /**
     * @param element element name
     * @param attr attribute name
     * @param value unexpected value
     */
    XMLUnexpectedAttributeValueException(const std::string& element, const std::string& attr, const std::string& value)
    : Exception("Attribute '%2%' for element '%1%' has unexpected value '%3%'", element, attr, value) {}
};

/**
 * Exceptions of this class are thown if two optional attributes in XML conflict with each other.
 */
struct XMLConflictingAttributesException: public Exception {
    /**
     * @param element element name
     * @param attr1 first attribute name
     * @param attr2 second attribute name
     */
    XMLConflictingAttributesException(const std::string& element, const std::string& attr1, const std::string& attr2)
    : Exception("Conflicting attributes '%2%' and '%3%' in element %1%", element, attr1, attr2) {}
};


//-------------- Connected with geometry: -----------------------

/**
 * Exceptions of this class are thrown when modules don't have geometry set
 */
struct NoGeometryException: public Exception {
    NoGeometryException(const std::string& where): Exception("$1$: No geometry specified", where) {}
};

/**
 * Exceptions of this class are thrown by some geometry element classes when there is no required child.
 */
struct NoChildException: public Exception {
    NoChildException(): Exception("Incomplete geometry tree") {}
};

/**
 * Exceptions of this class are thrown by some geometry element classes when there is no required child.
 */
struct NotUniqueElementException: public Exception {
    NotUniqueElementException(): Exception("Unique element instance required") {}
    NotUniqueElementException(const std::string msg): Exception(msg) {}
};

/**
 * Exceptions of this class are thrown when called operation on geometry graph will cause cyclic reference.
 */
struct CyclicReferenceException: public Exception {
    CyclicReferenceException(): Exception("Detected cycle in geometry tree") {}
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
 * Exceptions of this class are thrownby some geometry element classes when there is no required child.
 */
struct GeometryElementNamesConflictException: public Exception {

    /**
     * @param element_name name of element which is already exists
     */
    GeometryElementNamesConflictException(const std::string& element_name): Exception("Geometry element with given name \"" + element_name + "\" already exists") {}
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
    UnexpectedGeometryElementTypeException(): Exception("Geometry element has unexpected type") {}
};

/**
 * This exception is thrown when axis (typically with given name) is not found.
 */
struct NoSuchAxisNames: public Exception {
    //std::string materialName;

    /// @param axis_names name of axis which not exists
    NoSuchAxisNames(const std::string& axis_names)
        : Exception("No such axis names \"%1%\"", axis_names) {}
};


//-------------- Connected with meshes: -----------------------

/**
 * Exceptions of this class are thrown when modules don't have mesh set
 */
struct NoMeshException: public Exception {
    NoMeshException(const std::string& where): Exception("$1$: No mesh specified", where) {}
};


/**
 * This exception is thrown when the mesh is somehow bad
 */
struct BadMesh: public Exception {

    /**
     * @param where name of class/function/operation doing the computations
     * @param msg error message (format)
     * @param params paramters for @p msg
     */
    template <typename... Params>
    BadMesh(const std::string& where, const std::string& msg, Params... params)
        : Exception("%1%: Bad mesh: %2%", where, format(msg, params...)) {};
};




} // namespace plask

#endif  //PLASK__EXCEPTIONS_H
