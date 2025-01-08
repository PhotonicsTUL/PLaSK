/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "primitives.hpp"

#include <algorithm>

namespace plask {

static inline void ensureLo(double& to_be_lo, double how_lo) {
    if (how_lo < to_be_lo) to_be_lo = how_lo;
}

static inline void ensureHi(double& to_be_hi, double how_hi) {
    if (how_hi > to_be_hi) to_be_hi = how_hi;
}

//-------------  ---------------------

bool Box2D::operator ==(const Box2D &r) const {
    return lower == r.lower && upper == r.upper;
}

bool Box2D::operator !=(const Box2D &r) const {
    return lower != r.lower || upper != r.upper;
}

void Box2D::fix() {
    if (lower.c0 > upper.c0) std::swap(lower.c0, upper.c0);
    if (lower.c1 > upper.c1) std::swap(lower.c1, upper.c1);
}


bool Box2D::contains(const Vec<2, double >& p) const {
    return lower.c0 <= p.c0 && p.c0 <= upper.c0 &&
           lower.c1 <= p.c1 && p.c1 <= upper.c1;
}

bool Box2D::intersects(const plask::Box2D& other) const {
    return !(
        lower.c0 > other.upper.c0 ||
        lower.c1 > other.upper.c1 ||
        upper.c0 < other.lower.c0 ||
        upper.c1 < other.lower.c1
    );
}

void Box2D::makeInclude(const Vec<2, double >& p) {
    if (p.c0 < lower.c0) lower.c0 = p.c0; else ensureHi(upper.c0, p.c0);
    if (p.c1 < lower.c1) lower.c1 = p.c1; else ensureHi(upper.c1, p.c1);
}

void Box2D::makeInclude(const plask::Box2D& other) {
    ensureLo(lower.c0, other.lower.c0);
    ensureLo(lower.c1, other.lower.c1);
    ensureHi(upper.c0, other.upper.c0);
    ensureHi(upper.c1, other.upper.c1);
}

void Box2D::makeIntersection(const Box2D &other) {
    ensureHi(lower.c0, other.lower.c0);
    ensureHi(lower.c1, other.lower.c1);
    ensureLo(upper.c0, other.upper.c0);
    ensureLo(upper.c1, other.upper.c1);
}

Box2D Box2D::intersection(Box2D other) const {
    other.makeIntersection(*this);
    return other;
}

Box2D Box2D::extension(Box2D other) const {
    other.makeInclude(*this);
    return other;
}

Vec<2,double> Box2D::moveInside(Vec<2,double> p) const {
    if (p.c0 < lower.c0) p.c0 = lower.c0; else ensureLo(p.c0, upper.c0);
    if (p.c1 < lower.c1) p.c1 = lower.c1; else ensureLo(p.c1, upper.c1);
    return p;
}


//-------------  ---------------------

bool Box3D::operator ==(const Box3D &r) const {
    return lower == r.lower && upper == r.upper;
}

bool Box3D::operator !=(const Box3D &r) const {
    return lower != r.lower || upper != r.upper;
}

void Box3D::fix() {
    if (lower.c0 > upper.c0) std::swap(lower.c0, upper.c0);
    if (lower.c1 > upper.c1) std::swap(lower.c1, upper.c1);
    if (lower.c2 > upper.c2) std::swap(lower.c2, upper.c2);
}


bool Box3D::contains(const Vec<3, double >& p) const {
    return lower.c0 <= p.c0 && p.c0 <= upper.c0 &&
           lower.c1 <= p.c1 && p.c1 <= upper.c1 &&
           lower.c2 <= p.c2 && p.c2 <= upper.c2;
}

bool Box3D::intersects(const plask::Box3D& other) const {
    return !(
        lower.c0 > other.upper.c0 ||
        lower.c1 > other.upper.c1 ||
        lower.c2 > other.upper.c2 ||
        upper.c0 < other.lower.c0 ||
        upper.c1 < other.lower.c1 ||
        upper.c2 < other.lower.c2
    );
}

void Box3D::makeInclude(const Vec<3, double >& p) {
    if (p.c0 < lower.c0) lower.c0 = p.c0; else ensureHi(upper.c0, p.c0);
    if (p.c1 < lower.c1) lower.c1 = p.c1; else ensureHi(upper.c1, p.c1);
    if (p.c2 < lower.c2) lower.c2 = p.c2; else ensureHi(upper.c2, p.c2);
}

void Box3D::makeInclude(const plask::Box3D& other) {
    ensureLo(lower.c0, other.lower.c0);
    ensureLo(lower.c1, other.lower.c1);
    ensureLo(lower.c2, other.lower.c2);
    ensureHi(upper.c0, other.upper.c0);
    ensureHi(upper.c1, other.upper.c1);
    ensureHi(upper.c2, other.upper.c2);
}

void Box3D::makeIntersection(const Box3D &other) {
    ensureHi(lower.c0, other.lower.c0);
    ensureHi(lower.c1, other.lower.c1);
    ensureHi(lower.c2, other.lower.c2);
    ensureLo(upper.c0, other.upper.c0);
    ensureLo(upper.c1, other.upper.c1);
    ensureLo(upper.c2, other.upper.c2);
}

Box3D Box3D::extension(Box3D other) const {
    other.makeInclude(*this);
    return other;
}

Box3D Box3D::intersection(Box3D other) const {
    other.makeIntersection(*this);
    return other;
}

Vec<3,double> Box3D::moveInside(Vec<3,double> p) const {
    if (p.c0 < lower.c0) p.c0 = lower.c0; else ensureLo(p.c0, upper.c0);
    if (p.c1 < lower.c1) p.c1 = lower.c1; else ensureLo(p.c1, upper.c1);
    if (p.c2 < lower.c2) p.c2 = lower.c2; else ensureLo(p.c2, upper.c2);
    return p;
}

PLASK_API const char* const DIRECTION_NAMES[] = { "longitudinal", "transverse", "vertical" };

const Primitive<1>::DVec Primitive<1>::ZERO_VEC = 0.0;
const Primitive<2>::DVec Primitive<2>::ZERO_VEC = Vec<2>(0.0, 0.0);
const Primitive<3>::DVec Primitive<3>::ZERO_VEC = Vec<3>(0.0, 0.0, 0.0);

const Primitive<2>::DVec Primitive<2>::NAN_VEC = Vec<2>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
const Primitive<3>::DVec Primitive<3>::NAN_VEC = Vec<3>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

const Primitive<2>::Box Primitive<2>::INF_BOX = Box2D(- std::numeric_limits<double>::infinity(), - std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
const Primitive<3>::Box Primitive<3>::INF_BOX = Box3D(- std::numeric_limits<double>::infinity(), - std::numeric_limits<double>::infinity(), - std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
}
