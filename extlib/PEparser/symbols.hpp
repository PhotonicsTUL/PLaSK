#ifndef SYMBOLS_HPP
#define SYMBOLS_HPP

namespace dbg {

/**
 * Search for symbol under @p addres.
 * @param addres[in]
 * @param fun_name[out]
 * @param module_name[out]
 * @return @c true when search was success, only than @p fun_name and @p module_name are set.
 */
bool mingw_lookup(const void *addres, const char*& fun_name, const char*& module_name);

}

#endif // SYMBOLS_HPP
