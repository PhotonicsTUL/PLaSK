/**
 * \file Trivial license system.
 * This is a tivial license checking system. In future it should be replaced with some better
 * solution, which allows different licensing schemas and also allows to limit access to particular
 * solvers.
 */
#ifndef PLASK__LICENSE_DATA
#define PLASK__LICENSE_DATA

#include <ctime>
#include <vector>
#include <string>

namespace plask {

/**
 * Type of a license
 */
enum LicenseType {
    LICENSE_PERSONAL,
    LICENSE_NODE_LOCKED,
    LICENSE_FLOATING
};

#define READ_LICENSE_FIELD(Type) *reinterpret_cast<const Type*>(bytes); bytes += sizeof(Type)

/**
 * Basic Licensing information.
 * Specific license types should derive fromt this class.
 */
struct LicenseBase {
    LicenseType type;                ///< License type
    std::size_t number;              ///< License number
    std::time_t expire;              ///< License expiration time
    //std::vector<size_t> products;    ///< List of licensed products/solvers hashes

    static const char* deserialize(LicenseBase& dest, const char* bytes) {
        dest.type = READ_LICENSE_FIELD(LicenseType);
        dest.number = READ_LICENSE_FIELD(std::size_t);
        dest.expire = READ_LICENSE_FIELD(std::time_t);
        bytes += sizeof(std::size_t); //TODO
        return bytes;
    }
};

/**
 * Additional data for personal license.
 */
struct LicensePersonal: public LicenseBase {
    std::string user;                               ///< Name of the end user
    std::string email;                              ///< End user email

    static const char* deserialize(LicensePersonal& dest, const char* bytes) {
        bytes = LicenseBase::deserialize(dest, bytes);
        dest.user = *bytes; bytes += dest.user.size() + 1;
        dest.email = *bytes; bytes += dest.email.size() + 1;
        return bytes;
    }
};

}

#undef READ_LICENSE_FIELD

#endif // PLASK__LICENCSE_DATA
