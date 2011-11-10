

namespace plask {

/**
Base class for all exceptions throwed by plask library.
*/
struct Exception: public std::runtime_error {
    Exception(const std::string& msg): std::runtime_error(msg) {}
};

/**
This exce[tion is throw when some method is not implemented.
*/
struct NotImplemented: public Exception {
    std::string methodName;

    NotImplemented(const std::string& method_name)
    : Exception("Method not implemented: " + method_name), methodName(method_name) {}
};

struct NotSuchMaterial: public Exception {
    std::string materialName;

    NotSuchMaterial(const std::string& material_name)
    : Exception("No such material " + material_name), materialName(material_name) {}
};

} // namespace plask
