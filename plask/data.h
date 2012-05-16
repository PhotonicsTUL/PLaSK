#ifndef PLASK__DATA_H
#define PLASK__DATA_H

namespace plask {

/**
 * Store pointer and size.
 *
 * Has reference counter, and also can referee to external data (without management it).
 */
template <typename T>
class DataVector {

    std::size_t size_;                  ///< size of the stored data
    unsigned* gc_;                      ///< the reference count for the garbage collector
    T* data_;                           ///< The data of the matrix

    // Decrease GC counter and free memory if necessary
    void free() {
        if (gc_ && --(*gc_) == 0) {
            delete gc_;
            delete[] data_;
        }
    }

    void inc_ref() {
        if (gc_) ++(*gc_);
    }

  public:

    typedef T type;                     ///< type of the stored data

    typedef T* iterator;                ///< iterator type for the array
    typedef const T* const_iterator;    ///< constant iterator type for the array

    /// Create empty.
    DataVector() : size_(0), gc_(nullptr), data_(nullptr) {}

    /**
     * Create vector of given @p size.
     *
     * Reserve memory using new T[size] call.
     * @param size total size of the data
     */
    DataVector(std::size_t size) : size_(size), gc_(new unsigned(1)), data_(new T[size]) {}

    /**
     * Copy constructor. Only makes shallow copy (doesn't copy data).
     * @param src data source
     */
    DataVector(const DataVector& src): size_(src.size_), gc_(src.gc_), data_(src.data_) { inc_ref(); }

    /**
     * Assign operator. Only makes shallow copy (doesn't copy data).
     * @param M data source
     * @return *this
     */
    DataVector<T>& operator=(const DataVector<T>& M) {
        if (this == &M) return;
        this->free();
        size_ = M.size_;
        data_ = M.data_;
        gc_ = M.gc_;
        inc_ref();
        return *this;
    }

    /**
     * Move constructor.
     * @param src data to move
     */
    DataVector(DataVector&& src): size_(src.size_), gc_(src.gc_), data_(src.data_) {
        src.gc_ = 0;
        src.data_ = 0;
    }

    /**
     * Move operator.
     * @param src data source
     * @return *this
     */
    DataVector<T>& operator=(DataVector&& src) {
        this->free();
        size_ = src.size_;
        data_ = src.data_;
        gc_ = src.gc_;
        src.data_ = 0;
        src.gc_ = 0;
        return *this;
    }

    /**
     * Create vector out of existing data.
     * @param size  total size of the existing data
     * @param existing_data pointer to existing data
     * @param manage indicates whether the matrix object should manage the data and garbage-collect it (with delete[] operator)
     */
    DataVector(T* existing_data, std::size_t size, bool manage = false)
        : size_(size), gc_(manage ? new unsigned(1) : nullptr), data_(existing_data) {}

    DataVector(std::initializer_list<T> init): size_(init.size()), gc_(new unsigned(1)), data_(new T[init.size()]) {
        std::copy(init.begin(), init.end(), data_);
    }

    DataVector(std::size_t size, const T& value): size_(size), gc_(new unsigned(1)), data_(new T[size]) {
        std::fill(begin(), end(), value);
    }

    ~DataVector() { free(); }

    /// @return iterator referring to the first element in this matrix
    const_iterator begin() const { return data_; }
    iterator begin() { return data_; }

    /// @return iterator referring to the past-the-end element in this matrix
    const_iterator end() const { return data_ + size_; }
    iterator end() { return data_ + size_; }

    /// @return total size of the matrix/vector
    std::size_t size() const { return size_; }

    /// @return constant pointer to data
    const T* data() const { return data_; }

    /// @return pointer to data
    T* data() { return data_; }

    /**
     * Return n-th element of the data.
     * @param n number of element to return
     */
    const T& operator [](std::size_t n) const { return data_[n]; }

    /**
     * Return reference to the n-th element of the data.
     * @param n number of element to return
     */
    T& operator [](std::size_t n) { return data_[n]; }

    /**
     * Make a deep copy of the data.
     * @return new object with manage copy of this data
     */
    DataVector<T> copy() const {
        T* new_data = new T[size_];
        std::copy(begin(), end(), new_data);
        return DataVector<T>(new_data, size_, true);
    }

    /**
     * Check if this is the only one owner of data.
     * @return @c true only if this is the only one owner of data
     */
    bool unique() const {
        return (gc_ != nullptr) && (*gc_ == 1);
    }

    /**
     * Make copy of data only if this is not the only one owner of it.
     * @return copy of this: shallow if unique() is @c true, deep if unique() is @c false
     */
    DataVector<T> claim() const {
        return unique() ? *this : copy();
    }

};


} // namespace plask

#endif // PLASK__DATA_H
