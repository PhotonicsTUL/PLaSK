#ifndef PLASK__LOG_ID_H
#define PLASK__LOG_ID_H

#include <cstdint>
#include <string>

#include <plask/config.h>

namespace plask {

/**
 * Get unique number.
 *
 * This function is threads-safe.
 * @return unique number
 */
PLASK_API std::uint64_t getUniqueNumber();

/**
 * Get unique string.
 *
 * This function is threads-safe.
 * @return lexical_cast<std::string>(getUniqueNumber())
 */
PLASK_API std::string getUniqueString();

}   // namespace plask

#endif // PLASK__LOG_ID_H
