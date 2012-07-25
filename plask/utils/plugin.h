#ifndef PLASK__UTILS_PLUGIN_H
#define PLASK__UTILS_PLUGIN_H

#if defined(_MSC_VER) || defined(__MINGW32__)
#define PLASK__UTILS_PLUGIN_WINAPI
#include <windows.h>
#else
#define PLASK__UTILS_PLUGIN_DLOPEN
#include <dlfcn.h>
#endif

#include <string>

namespace plask {

/**
 * Hold opened shared library. Portable, thin wrapper over system handler to library.
 *
 * Close holded library in destructor.
 */
struct DynamicLibrary {

    /// Type of system shared library handler.
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    typedef HINSTANCE handler_t;
#else
    typedef void* handler_t;
#endif

private:
    /// System shared library handler
    handler_t handler;

public:

    /**
     * Open library from file with given name @p filename.
     * @param filename name of file with library to load
     */
    explicit DynamicLibrary(const std::string &filename);

    /**
     * Don't open any library. You can call open later.
     */
    DynamicLibrary();

    /// Coping of libary is not alowed
    DynamicLibrary(const DynamicLibrary&) = delete;

    /// Coping of libary is not alowed
    DynamicLibrary& operator=(const DynamicLibrary &) = delete;

    /**
     * Move library ownership from @p to_move to this.
     * @param to_move source of moving library ownership
     */
    DynamicLibrary(DynamicLibrary&& to_move);

    /**
     * Move library ownership from @p to_move to this.
     *
     * Same as: <code>swap(to_move);</code>
     * @param to_move source of moving library ownership
     * @return *this
     */
    DynamicLibrary& operator=(DynamicLibrary && to_move) {
        swap(to_move);  //destructor of to_move will close current this library
        return *this;
    }

    /**
     * Swap library ownerships between this and to_swap.
     * @param to_swap library to swap with
     */
    void swap(DynamicLibrary& to_swap) {
        std::swap(handler, to_swap.handler);
    }

    /**
     * Dispose library.
     */
    ~DynamicLibrary();

    /**
     * Open library from file with given name @p filename.
     *
     * Close already opened library wrapped by this if any.
     * @param filename name of file with library to load
     */
    void open(const std::string &filename);

    /**
     * Close opened library.
     */
    void close();

    /**
     * Get symbol from library.
     *
     * Throw excpetion if library is not opened.
     * @param symbol_name name of symbol to get
     * @return symbol with given name, or @c nullptr if there is no symbol with given name
     */
    void* operator[](const std::string &symbol_name);

    /**
     * Get symbol from library.
     *
     * Throw excpetion if library is not opened or if there is no symbol with given name.
     * @param symbol_name name of symbol to get
     * @return symbol with given name
     */
    void* requireSymbol(const std::string &symbol_name);

    /**
     * Check if library is already open.
     * @return @c true only if library is already open
     */
    bool isOpened() const { return handler != 0; }

    /**
     * Get system handler.
     *
     * Type of result is system specyfic (DynamicLibrary::handler_t).
     * @return system handler
     */
    handler_t getSystemHandler() const { return handler; }

    /**
     * Release ownership over holded system library handler.
     * Don't close this library.
     */
    handler_t release();

};

}   // namespace plask

namespace std {
/// std::swap implementation for dynamic libraries
inline void swap(plask::DynamicLibrary& a, plask::DynamicLibrary& b) { a.swap(b); }
}

#endif // PLASK__UTILS_PLUGIN_H
