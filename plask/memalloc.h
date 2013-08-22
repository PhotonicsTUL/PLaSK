#ifndef PLASK__MEMALLOC_H
#define PLASK__MEMALLOC_H

#include <cstdlib>
#include <utility>
#include <limits>
#include <new>

namespace plask {

#if (defined(__GLIBC__) && ((__GLIBC__>=2 && __GLIBC_MINOR__ >= 8) || __GLIBC__>2) && defined(__LP64__)) || \
    defined(__APPLE__) || defined(_WIN64) || (defined(__FreeBSD__) && !defined(__arm__) && !defined(__mips__))
#   define PLASK_MALLOC_ALIGNED 1
#else
#   define PLASK_MALLOC_ALIGNED 0
#endif


#if !PLASK_MALLOC_ALIGNED && !defined(_MSC_VER)
namespace detail {

    /**
     * \internal Like malloc, but the returned pointer is guaranteed to be 16-byte aligned.
     */
    inline void* custom_aligned_malloc(std::size_t size)
    {
        void *original = std::malloc(size+16);
        if (original == 0) return 0;
        void *aligned = reinterpret_cast<void*>((reinterpret_cast<size_t>(original) & ~(size_t(15))) + 16);
        *(reinterpret_cast<void**>(aligned) - 1) = original;
        return aligned;
    }

    /**
     * \internal Free memory allocated with custom_aligned_malloc
     */
    inline void custom_aligned_free(void *ptr)
    {
        if (ptr) std::free(*(reinterpret_cast<void**>(ptr) - 1));
    }

    /**
     * \internal Reallocate aligned memory.
     */
    inline void* custom_aligned_realloc(void* ptr, std::size_t size, std::size_t=0)
    {
        if (ptr == 0) return custom_aligned_malloc(size);
        void *original = *(reinterpret_cast<void**>(ptr) - 1);
        original = std::realloc(original,size+16);
        if (original == 0) return 0;
        void *aligned = reinterpret_cast<void*>((reinterpret_cast<size_t>(original) & ~(size_t(15))) + 16);
        *(reinterpret_cast<void**>(aligned) - 1) = original;
        return aligned;
    }

}
#endif

/**
 * Allocate \a size bytes. The returned pointer is guaranteed to have 16 bytes alignment.
 * \param size number of bytes to allocate
 * \throws std::bad_alloc on allocation failure
 */
inline void* aligned_malloc(std::size_t size)
{
    void *result;
#if PLASK_MALLOC_ALIGNED
    result = std::malloc(size);
#elif defined _MSC_VER
    result = _aligned_malloc(size, 16);
#else
    result = detail::custom_aligned_malloc(size);
#endif
    if(!result && size) throw std::bad_alloc();
    return result;
}

/**
 * Free memory allocated with aligned_malloc.
 * \param ptr pointer to free
 */
inline void aligned_free(void *ptr)
{
#if PLASK_MALLOC_ALIGNED
    std::free(ptr);
#elif defined(_MSC_VER)
    _aligned_free(ptr);
#else
    detail::custom_aligned_free(ptr);
#endif
}

/**
 * Reallocate an aligned block of memory.
 * \param ptr pointer to reallocate
 * \param new_size new size
 * \param old_size old size
 * \throws std::bad_alloc on allocation failure
**/
inline void* aligned_realloc(void *ptr, std::size_t new_size, std::size_t old_size=0)
{
    void *result;
#if PLASK_MALLOC_ALIGNED
    result = std::realloc(ptr,new_size);
#elif defined(_MSC_VER)
    result = _aligned_realloc(ptr,new_size,16);
#else
    result = detail::custom_aligned_realloc(ptr,new_size,old_size);
#endif
    if (!result && new_size) throw std::bad_alloc();
    return result;
}

/**
 * Create new data with aligned allocation.
 * \tparam T object type
 * \param num number of array elements
 * \return pointer to reserved memory
 */
template <typename T>
inline T* aligned_malloc(std::size_t num=1) {
    T* mem = reinterpret_cast<T*>(aligned_malloc(num * sizeof(T)));
    return mem;
}

/**
 * Delete data with aligned allocation
 * \param ptr pointer of object to delete
 * \tparam T object type
 */
template <typename T>
inline void aligned_free(T* ptr) {
    aligned_free(reinterpret_cast<void*>(const_cast<typename std::remove_const<T>::type*>(ptr)));
}

/**
 * Aligned deleter for use e.g. in unique_ptr
 */
template <typename T>
struct aligned_deleter {
    constexpr aligned_deleter() noexcept = default;
    template<typename U, typename=typename std::enable_if<std::is_convertible<U*,T*>::value>::type>
        aligned_deleter(const aligned_deleter<U>&) noexcept {}

    void operator()(T* ptr) const {
        aligned_free<T>(ptr);
    }
};

/**
 * Create new object with aligned allocation
 * \tparam T object type
 * \param args arguments forwarded to object constructor
 * \return pointer to reserved memory
 */
template <typename T, typename... Args>
inline T* aligned_new(Args&&... args) {
    T* mem = reinterpret_cast<T*>(aligned_malloc(sizeof(T)));
    new(mem) T(std::forward<Args>(args)...);
    return mem;
}

/**
 * Delete object with aligned allocation
 * \param ptr pointer of object to delete
 * \tparam T object type
 */
template <typename T>
inline void aligned_delete(T* ptr) {
    ptr->~T();
    aligned_free(ptr);
}

/**
 * Create new array with aligned allocation
 * \tparam T object type
 * \param num number of array elements
 * \param args arguments forwarded to object constructor
 */
template <typename T, typename... Args>
inline T* aligned_new_array(std::size_t num, Args&&... args) {
    T* mem = reinterpret_cast<T*>(aligned_malloc(num * sizeof(T)));
    for (size_t i = 0; i != num; ++i) new(mem+i) T(std::forward<Args>(args)...);
    return mem;
}

/**
 * Delete array with aligned allocation
 * \param ptr pointer of object to delete
 * \param num number of array elements
 * \tparam T object type
 */
template <typename T>
inline void aligned_delete_array(std::size_t num, T* ptr) {
    while (num)
        ptr[--num].~T();
    aligned_free(ptr);
}


/**
 * STL compatible allocator to use with with 16 byte aligned types
 */
template<class T>
struct aligned_allocator {

    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;
    typedef T*              pointer;
    typedef const T*        const_pointer;
    typedef T&              reference;
    typedef const T&        const_reference;
    typedef T               value_type;

    template<class U>
    struct rebind { typedef aligned_allocator<U> other; };

    pointer address(reference value) const { return &value; }

    const_pointer address(const_reference value) const { return &value; }

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator&) {}

    template<class U>
    aligned_allocator(const aligned_allocator<U>&) {}

    ~aligned_allocator(){}

    size_type max_size() const { return (std::numeric_limits<size_type>::max)(); }

    pointer allocate(size_type num) { return aligned_malloc<T>(num); }

    void construct(pointer p, const T& value) { new(p) T(value); }

    template<typename... Args>
    void  construct(pointer p, Args&&... args) { new(p) T(std::forward<Args>(args)...); }

    void destroy(pointer p) { p->~T(); }

    void deallocate(pointer p, size_type) { aligned_free(p); }

    bool operator!=(const aligned_allocator<T>&) const { return false; }

    bool operator==(const aligned_allocator<T>&) const { return true; }
};



} // namespace plask


#endif // PLASK__MEMALLOC_H
