#ifndef PLASK__EXCEPTIONS_H
#define PLASK__EXCEPTIONS_H

/** @file
This file includes definitions of all exceptions classes which are used in PLaSK.
*/

#include <stdexcept>

namespace plask {

/**
 * Base class for all exceptions thrown by plask library.
 */
struct Exception: public std::runtime_error {
    Exception(const std::string& msg): std::runtime_error(msg) {}
};

/**
 * Exceptions of this class are throw in cases of critical and very unexpected errors (possible plask bugs).
 */
struct CriticalException: public Exception {
    CriticalException(const std::string& msg): Exception(msg) {}
};

/**
 * This exception is thrown when some method is not implemented.
 */
struct NotImplemented: public Exception {
    //std::string methodName;

    NotImplemented(const std::string& method_name)
    : Exception("Method not implemented: " + method_name)/*, methodName(method_name)*/ {}

    NotImplemented(const std::string& where, const std::string& method_name)
    : Exception("In " + where + ": Method not implemented: " + method_name)/*, methodName(method_name)*/ {}
};

/**
 * This exception is thrown, typically on access to @ref plask::Receiver "receiver" data,
 * when there is no @ref plask::Provider "provider" connected with it.
 * @see @ref providers
 */
struct NoProvider: public Exception {
    NoProvider(): Exception("No provider.") {}
};

/**
 * Exceptions of this class are throw by some geometry element classes when there is no required child.
 */
struct NoChildException: public Exception {
    NoChildException(): Exception("No child.") {}
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

    NoSuchMaterial(const std::string& material_name)
    : Exception("No such material " + material_name)/*, materialName(material_name)*/ {}
};

struct MaterialMethodNotImplemented: public NotImplemented {

    MaterialMethodNotImplemented(const std::string& material_name, const std::string& method_name)
    : NotImplemented("material " + material_name, method_name) {
    }

};

/**
 * Exceptions of this class are throw when material string parser find errors.
 */
struct MaterialParseException: public Exception {
    MaterialParseException(): Exception("Material parse error.") {}
    MaterialParseException(const std::string& msg): Exception("Material parse error: " + msg) {}
};


} // namespace plask

#endif  //PLASK__EXCEPTIONS_H
