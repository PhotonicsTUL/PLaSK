#include "primitives.h"

#include <algorithm>

namespace plask {

inline void ensureLo(double& to_be_lo, double how_lo) {
    if (how_lo < to_be_lo) to_be_lo = how_lo;
}

inline void ensureHi(double& to_be_hi, double how_hi) {
    if (how_hi > to_be_hi) to_be_hi = how_hi;
}

//------------- Rect2d ---------------------
    
void Rect2d::fix() {
    if (lower.x > upper.x) std::swap(lower.x, upper.x);
    if (lower.y > upper.y) std::swap(lower.y, upper.y);
}


bool Rect2d::inside(const Vec2< double >& p) const {
    return lower.x <= p.x && p.x <= upper.x &&
           lower.y <= p.y && p.y <= upper.y;    
}

bool Rect2d::intersect(const plask::Rect2d& other) const {
    return !(
        lower.x > other.upper.x ||
        lower.y > other.upper.y ||
        upper.x < other.lower.x ||
        upper.y < other.lower.y
    );
}

void Rect2d::include(const Vec2< double >& p) {
    if (p.x < lower.x) lower.x = p.x; else ensureHi(upper.x, p.x);
    if (p.y < lower.y) lower.y = p.y; else ensureHi(upper.y, p.y);
}

void Rect2d::include(const plask::Rect2d& other) {
    ensureLo(lower.x, other.lower.x);
    ensureLo(lower.y, other.lower.y);
    ensureHi(upper.x, other.upper.x);
    ensureHi(upper.y, other.upper.y);
}

//------------- Rect3d ---------------------

void Rect3d::fix() {
    if (lower.x > upper.x) std::swap(lower.x, upper.x);
    if (lower.y > upper.y) std::swap(lower.y, upper.y);
    if (lower.z > upper.z) std::swap(lower.z, upper.z);
}


bool Rect3d::inside(const Vec3< double >& p) const {
    return lower.x <= p.x && p.x <= upper.x &&
           lower.y <= p.y && p.y <= upper.y &&
           lower.z <= p.z && p.z <= upper.z;
}

bool Rect3d::intersect(const plask::Rect3d& other) const {
    return !(
        lower.x > other.upper.x ||
        lower.y > other.upper.y ||
        lower.z > other.upper.z ||
        upper.x < other.lower.x ||
        upper.y < other.lower.y ||
        upper.z < other.lower.z
    );
}

void Rect3d::include(const Vec3< double >& p) {
    if (p.x < lower.x) lower.x = p.x; else ensureHi(upper.x, p.x);
    if (p.y < lower.y) lower.y = p.y; else ensureHi(upper.y, p.y);
    if (p.z < lower.z) lower.z = p.z; else ensureHi(upper.z, p.z);
}

void Rect3d::include(const plask::Rect3d& other) {
    ensureLo(lower.x, other.lower.x);
    ensureLo(lower.y, other.lower.y);
    ensureLo(lower.z, other.lower.z);
    ensureHi(upper.x, other.upper.x);
    ensureHi(upper.y, other.upper.y);
    ensureHi(upper.z, other.upper.z);
}

}
