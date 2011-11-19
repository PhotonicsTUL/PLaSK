#ifndef PLASK__EXCEPTIONS_H
#define PLASK__EXCEPTIONS_H

#include <stdexcept>

namespace plask {

/**
 * Base class for all exceptions throwed by plask library.
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
This exce[tion is throw when some method is not implemented.
*/
struct NotImplemented: public Exception {
    //std::string methodName;

    NotImplemented(const std::string& method_name)
    : Exception("Method not implemented: " + method_name)/*, methodName(method_name)*/ {}
};

/**
 * This excpetion is throw when material (typically with given name) is not found.
 */
struct NoSuchMaterial: public Exception {
    //std::string materialName;

    NoSuchMaterial(const std::string& material_name)
    : Exception("No such material " + material_name)/*, materialName(material_name)*/ {}
};

struct NoProvider: public Exception {
    NoProvider(): Exception("No provider.") {}
};

} // namespace plask

#endif  //PLASK__EXCEPTIONS_H
