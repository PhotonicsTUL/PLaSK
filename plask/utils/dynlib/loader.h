#ifndef PLASK__UTILS_DYNLIB_LOADER_H
#define PLASK__UTILS_DYNLIB_LOADER_H

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define PLASK__UTILS_PLUGIN_WINAPI
#else
#   define PLASK__UTILS_PLUGIN_DLOPEN
#endif

#include <string>
#include <functional>   //std::hash

#include <plask/config.h>

namespace plask {

/**
 * Hold opened shared library. Portable, thin wrapper over system handle to library.
 *
 * Close holded library in destructor.
 */
struct PLASK_API DynamicLibrary {

    enum Flags {
        DONT_CLOSE = 1  ///< if this flag is set DynamicLibrary will not close the library, but it will be closed on application exit
    };

    typedef void* handle_t;    // real type on windows is HINSTANCE, but it is a pointer (to struct), so we are going to cast it from/to void* to avoid indluding windows.h

private:
    /// System shared library handle
    handle_t handle;
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    bool unload;    // true if lib. should be unloaded, destructor just don't call FreeLibrary if this is false, undefined when handle == 0
#endif

public:

    /**
     * Get system handler to the loaded library.
     *
     * In windows it can be casted to HINSTANCE (defined in windows.h).
     * @return the system handler
     */
    handle_t getHandle() const { return handle; }

    static constexpr const char* DEFAULT_EXTENSION =
#ifdef PLASK__UTILS_PLUGIN_WINAPI
            ".dll";
#else
            ".so";
#endif

    /**
     * Open library from file with given name @p filename.
     * @param filename name of file with library to load
     * @param flags flags which describes configuration of open/close process, one or more (or-ed) flags from DynamicLibrary::Flags set
     */
    explicit DynamicLibrary(const std::string& filename, unsigned flags);

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
    DynamicLibrary(DynamicLibrary&& to_move) noexcept;

    /**
     * Move library ownership from @p to_move to this.
     *
     * Same as: <code>swap(to_move);</code>
     * @param to_move source of moving library ownership
     * @return *this
     */
    DynamicLibrary& operator=(DynamicLibrary && to_move) noexcept {
        swap(to_move);  // destructor of to_move will close current this library
        return *this;
    }

    /**
     * Swap library ownerships between this and to_swap.
     * @param to_swap library to swap with
     */
    void swap(DynamicLibrary& to_swap) noexcept {
        std::swap(handle, to_swap.handle);
#ifdef PLASK__UTILS_PLUGIN_WINAPI
        std::swap(unload, to_swap.unload);
#endif
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
     * @param flags flags which describe configuration of open/close process, one or more (or-ed) flags from DynamicLibrary::Flags set
     */
    void open(const std::string& filename, unsigned flags);

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
    void* getSymbol(const std::string &symbol_name) const;

    /**
     * Get symbol from library and cast it to given type.
     *
     * Throw excpetion if library is not opened.
     * @param symbol_name name of symbol to get
     * @return symbol with given name casted to given type, or @c nullptr if there is no symbol with given name
     * @tparam SymbolType required type to which symbol will be casted
     */
    template <typename SymbolType>
    SymbolType getSymbol(const std::string &symbol_name) const {
        return reinterpret_cast<SymbolType>(getSymbol(symbol_name));
    }

    /// Same as getSymbol(const std::string &symbol_name)
    void* operator[](const std::string &symbol_name) const {
        return getSymbol(symbol_name);
    }

    /**
     * Get symbol from library.
     *
     * Throw excpetion if library is not opened or if there is no symbol with given name.
     * @param symbol_name name of symbol to get
     * @return symbol with given name
     */
    void* requireSymbol(const std::string &symbol_name) const;

    /**
     * Get symbol from library and cast it to given type.
     *
     * Throw excpetion if library is not opened or if there is no symbol with given name.
     * @param symbol_name name of symbol to get
     * @return symbol with given name, casted to given type
     * @tparam SymbolType required type to which symbol will be casted
     */
    template <typename SymbolType>
    SymbolType requireSymbol(const std::string &symbol_name) const {
        return reinterpret_cast<SymbolType>(requireSymbol(symbol_name));
    }

    /**
     * Check if library is already open.
     * @return @c true only if library is already open
     */
    bool isOpen() const { return handle != 0; }

    /**
     * Get system handle.
     *
     * Type of result is system specyfic (DynamicLibrary::handle_t).
     * @return system handle
     */
    handle_t getSystemHandler() const { return handle; }

    /**
     * Release ownership over holded system library handle.
     * This does not close the library.
     * @return system library handle which ownership has been relased
     *          (on windows it can be casted to HINSTANCE)
     */
    handle_t release();

    /**
     * Compare operator, defined to allow store dynamic libriaries in standard containers which require this.
     */
    bool operator<(const DynamicLibrary& other) const {
#ifdef PLASK__UTILS_PLUGIN_WINAPI
        return this->handle < other.handle || (this->handle == other.handle && !this->unload && other.unload);
#else
        return this->handle < other.handle;
#endif
    }

    /**
     * Compare operator, defined to allow store dynamic libriaries in standard containers which require this.
     */
    bool operator == (const DynamicLibrary& other) const {
#ifdef PLASK__UTILS_PLUGIN_WINAPI
        return (this->handle == other.handle) && (this->unload == other.unload);
#else
        return this->handle == other.handle;
#endif
    }

};

}   // namespace plask

namespace std {

/// std::swap implementation for dynamic libraries
inline void swap(plask::DynamicLibrary& a, plask::DynamicLibrary& b) noexcept { a.swap(b); }

/// hash method, allow to store dynamic libraries in hash maps
template<>
struct hash<plask::DynamicLibrary> {
    std::hash<plask::DynamicLibrary::handle_t> h;
public:
    std::size_t operator()(const plask::DynamicLibrary &s) const {
        return h(s.getSystemHandler());
    }
};



}

#endif // PLASK__UTILS_DYNLIB_LOADER_H
