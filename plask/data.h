#ifndef PLASK__DATA_H
#define PLASK__DATA_H

namespace plask {


/**
 * Base class for every matrix and vector.
 *
 * It provides memory management and all the common operations.
 */
template <typename T>
class DataVector {

  protected:

    std::size_t size_;                  ///< size of the stored data
    unsigned* gc_;                      ///< the reference count for the garbage collector
    T* data_;                           ///< The data of the matrix

    // Decrease GC counter and free memory if necessary
    void free() {
        if (gc_ && --(*gc_) == 0)
            delete gc_; delete[] data_;
    }

  public:

    typedef T type;                     ///< type of the stored data

    typedef T* iterator;                ///< iterator type for the array
    typedef const T* const_iterator;    ///< constant iterator type for the array

    /// Create empty matrix/vector
    DataVector() : size_(0), gc_(NULL), data_(NULL) {}

    /**
     * Create vector of given size filled with zeros.
     * \param size total size of the matrix/vector
     */
    DataVector(std::size_t size) : size_(size), gc_(new unsigned), data_(new T[size]) {
        *gc_ = 1;
    }

    /**
     * Copy constructor. Only makes shallow copy.
     * \param src source matrix
     */
    DataVector(const DataVector& src) : size_(src.size_), gc_(src.gc_), data_(src.data_) { if (gc_) ++(*gc_); }

    /**
     * Create vector out of existing data.
     * \param size  total size of the matrix/vector
     * \param existing_data pointer to matrix data
     * \param manage indicates whether the matrix object should manage the data and garbage-collect it (with delete[] operator)
     */
    DataVector(std::size_t size, T* existing_data, bool manage=false) : size_(size), gc_(NULL) {
        data_ = existing_data;
        if (manage) {
            gc_ = new unsigned;
            *gc_ = 1;
        }
    }

    ~DataVector() { free(); }

    /// \return iterator referring to the first element in this matrix
    const_iterator begin() const { return data_; }

    /// \return iterator referring to the past-the-end element in this matrix
    const_iterator end() const { return data_ + size_; }

    /// \return total size of the matrix/vector
    std::size_t size() const { return size_; }

    /// \return constant pointer to data
    const T* data() const { return data_; }

    /// \return pointer to data
    T* data() { return data_; }

    /**
     * Return n-th element of the data_.
     * \param n number of element to return
     */
    const T& operator [](std::size_t n) const { return data[n]; }

    /**
     * Return reference to the n-th element of the data_.
     * \param n number of element to return
     */
    T& operator [](std::size_t n) { return data[n]; }

    DataVector<T>& operator=(const DataVector<T>& M) {
        this->free();
        size_ = M.size_; data_ = M.data_; gc_ = M.gc_; (*gc_)++;
        return *this;
    }

    /**
     * Make a deep copy of the matrix.
     * \return new matrix with data copied to the new location
     */
    DataVector<T> copy() const {
        T* new_data = new T[size_];
        std::copy(data_, data_ + size_, new_data);
        return DataVector<T>(size_, new_data, true);
    }

};


} // namespace plask

#endif // PLASK__DATA_H