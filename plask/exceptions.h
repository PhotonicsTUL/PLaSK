#ifndef PLASK__EXCEPTIONS_H
#define PLASK__EXCEPTIONS_H

/** @file
This file contains definitions of most exceptions classes which are used in PLaSK.
*/

#include <stdexcept>
#include "utils/format.h"
#include "utils/string.h"

#include <plask/config.h>

namespace plask {

/**
 * Base class for all exceptions thrown by plask library.
 */
struct PLASK_API Exception: public std::runtime_error {

    /// @param msg error message
    Exception(const std::string& msg);

    /**
     * Format error message using boost::format.
     */
    template <typename... T>
    Exception(const std::string& msg, const T&... args): Exception(format(msg, args...)) {}
};

/**
 * Exceptions of this class are thrownin cases of critical and very unexpected errors (possible plask bugs).
 */
struct PLASK_API CriticalException: public Exception {

    /// @param msg error message
    CriticalException(const std::string& msg): Exception("Critical exception: " + msg) {}

    template <typename... T>
    CriticalException(const std::string& msg, const T&... args): Exception("Critical exception: " + msg, args...) {}
};

/**
 * This exception is thrown when some method is not implemented.
 */
struct PLASK_API NotImplemented: public Exception {
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
struct PLASK_API OutOfBoundsException: public Exception {

    OutOfBoundsException(const std::string& where, const std::string& argname)
        : Exception("%1%: argument %2% out of bounds", where, argname) {}

    template <typename BoundTypeWas, typename BoundTypeLo, typename BoundTypeHi>
    OutOfBoundsException(const std::string& where, const std::string& argname, const BoundTypeWas& was, const BoundTypeLo& lo, const BoundTypeHi& hi)
        : Exception("%1%: argument %2% out of bounds, should be between %3% and %4%, but was %5%", where, argname, lo, hi, was) {}
};

/**
 * This exception is thrown if there is a problem with dimensions.
 */
struct PLASK_API DimensionError: public Exception {
    template <typename... T>
    DimensionError(T... args) : Exception(args...) {}
};

/**
 * This exception is thrown when value specified by the user is bad
 */
struct PLASK_API BadInput: public Exception {

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
 * This exception is called when operation on data vectors cannot be performed
 */
struct PLASK_API DataError: public Exception {

    /**
     * @param msg error message (format)
     * @param params formating parmeters for msg
     */
    template <typename... Params>
    DataError(const std::string& msg, Params... params)
        : Exception("%1%", format(msg, params...)) {};
};


/**
 * This exception should be thrown by solvers in case of error in computations.
 */
struct PLASK_API ComputationError: public Exception {

    /**
     * @param where name of class/function/operation doing the computations
     * @param msg error message
     * @param params formating parmeters for msg
     */
    template <typename... Params>
    ComputationError(const std::string& where, const std::string& msg, Params... params)
        : Exception("%1%: %2%", where, format(msg, params...)) {};
};

/**
 * This is throwed if name is bad id.
 */
struct PLASK_API BadId: public Exception {

    BadId(const std::string& where, const char* str_to_check, char underline_ch = '_')
        : Exception("\"%1%\" is bad name for %2%, this name shouldn't be empty and should consists of english letters, '%3%' character and digits (but not at beginning).", str_to_check, where, underline_ch) {};

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
struct PLASK_API NoProvider: public Exception {
    NoProvider(const char* provider_name): Exception("No provider nor value for %1%", provider_name) {}
};

struct PLASK_API NoValue: public Exception {
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
class PLASK_API NoSuchMaterial: public Exception {

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
struct PLASK_API MaterialMethodNotImplemented: public NotImplemented {

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
struct PLASK_API MaterialMethodNotApplicable: public Exception {

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
struct PLASK_API MaterialParseException: public Exception {
    MaterialParseException(): Exception("Material parse error") {}
    /// @param msg error message
    MaterialParseException(const std::string& msg): Exception(msg) {}

    template <typename... T>
    MaterialParseException(const std::string& msg, const T&... args): Exception(msg, args...) {
    }
};

class PLASK_API MaterialCantBeMixedException: public Exception {

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


//-------------- Connected with geometry: -----------------------

/**
 * Exceptions of this class are thrown when solvers don't have geometry set
 */
struct PLASK_API NoGeometryException: public Exception {
    NoGeometryException(const std::string& where): Exception("%1%: No geometry specified", where) {}
};

/**
 * Exceptions of this class are thrown by some geometry object classes when there is no required child.
 */
struct PLASK_API NoChildException: public Exception {
    NoChildException(): Exception("Incomplete geometry tree") {}
};

/**
 * Exceptions of this class are thrown by some geometry object classes
 */
struct PLASK_API NotUniqueObjectException: public Exception {
    NotUniqueObjectException(): Exception("Unique object instance required") {}
    NotUniqueObjectException(const std::string& msg): Exception(msg) {}

    template <typename... T>
    NotUniqueObjectException(const std::string& msg, const T&... args): Exception(msg, args...) {}
};

/**
 * Exceptions of this class are thrown when called operation on geometry graph will cause cyclic reference.
 */
struct PLASK_API CyclicReferenceException: public Exception {
    CyclicReferenceException(): Exception("Detected cycle in geometry tree") {}
};

/**
 * This exception is thrown when geometry object (typically with given name) is not found.
 */
struct PLASK_API NoSuchGeometryObjectType: public Exception {
    //std::string materialName;

    /**
     * @param object_type_name name of object type which is not found
     */
    NoSuchGeometryObjectType(const std::string& object_type_name)
        : Exception("No geometry object with given type name \"" + object_type_name + "\"")/*, materialName(material_name)*/ {}
};

/**
 * Exceptions of this class are thrownby some geometry object classes when there is no required child.
 */
struct PLASK_API NamesConflictException: public Exception {

    /**
     * @param what type of object
     * @param object_name name of object which is already exists
     */
    NamesConflictException(const std::string& what, const std::string& object_name):
        Exception(what + " with name \"" + object_name + "\" already exists.") {}
};

/**
 * This exception is thrown when geometry object (typically with given name) is not found.
 */
struct PLASK_API NoSuchGeometryObject: public Exception {
    //std::string materialName;

    /**
     * @param object_name name of object which is not found
     */
    NoSuchGeometryObject(const std::string& object_name)
    : Exception("No geometry object with name \"" + object_name + "\"") {}
};

/**
 * This exception is thrown when geometry (typically with given name) is not found.
 */
struct PLASK_API NoSuchGeometry: public Exception {
    /**
     * @param object_name name of object which is not found
     */
    NoSuchGeometry(const std::string& object_name)
    : Exception("No geometry of required type with name \"" + object_name + "\"") {}
};

/**
 * This exception is thrown when named PatHints are not found.
 */
struct PLASK_API NoSuchPath: public Exception {
    /**
     * @param object_name name of object which is not found
     */
    NoSuchPath(const std::string& object_name)
    : Exception("No path with name \"" + object_name + "\"") {}
};

/**
 * This exception is thrown when geometry object has type different than expectation (for example is 3d but expected 2d).
 */
struct PLASK_API UnexpectedGeometryObjectTypeException: public Exception {
    UnexpectedGeometryObjectTypeException(): Exception("Geometry object has unexpected type") {}
};

/**
 * This exception is thrown when axis (typically with given name) is not found.
 */
struct PLASK_API NoSuchAxisNames: public Exception {
    //std::string materialName;

    /// @param axis_names name of axis which not exists
    NoSuchAxisNames(const std::string& axis_names)
        : Exception("No such axis names \"%1%\"", axis_names) {}
};


//-------------- Connected with meshes: -----------------------

/**
 * Exceptions of this class are thrown when solvers don't have mesh set
 */
struct PLASK_API NoMeshException: public Exception {
    NoMeshException(const std::string& where): Exception("$1$: No mesh specified", where) {}
};


/**
 * This exception is thrown when the mesh is somehow bad
 */
struct PLASK_API BadMesh: public Exception {

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
