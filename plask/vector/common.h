#ifndef PLASK__VECTOR__COMMON_H
#define PLASK__VECTOR__COMMON_H

/** @file
In this file some basis common for all vectors (2D and 3D) are defined.
*/

namespace plask {


/// Generic template for 2D and 3D vectors
template <int dim, typename T=double>
struct Vec {};


/**
 * Vector component helper.
 * This class allow to perform operations on single components of vectors.
 */
template <int dim, typename T, int i>
struct VecComponent {

    static const int DIMS = dim;

    T c[dim];

    /**
     * Assign value to the component
     * \param val value to assign
     */
    VecComponent<dim,T,i>& operator=(const T& val) { c[i] = val; return *this; }

    /**
     * Extract value from the component
     */
    operator T() const { return c[i]; }
};


namespace axis {
    const std::size_t lon_index = 0;
    const std::size_t tran_index = 1;
    const std::size_t up_index = 2;
}   // axis

/*
 * Rotate @p r over up axis to lie on lon-tran plane.
 * @param r cuboid
 * @return rectangle
 */
/**static Block2D rotateToLonTran(const Block3D& r) {
    Box2D result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}*/

}   // namespace plask

#endif // PLASK__VECTOR__COMMON_H
