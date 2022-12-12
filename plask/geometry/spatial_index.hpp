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
#ifndef PLASK__GEOMETRY_SPATIAL_INDEX_H
#define PLASK__GEOMETRY_SPATIAL_INDEX_H

#include "transform.hpp"

namespace plask {

/**
 * Base class for cache and nodes of cache.
 *
 * It has some methods similar to this in GeometryObjectContainer API and is used by TranslationContainer.
 */
template <int DIMS>
struct SpatialIndexNode {

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const = 0;

    virtual bool contains(const Vec<DIMS>& p) const = 0;

    virtual GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const = 0;

    virtual ~SpatialIndexNode() {}
};


/**
 * Build spatial index.
 *
 * The index prevent reverse order of children in case of material searching.
 * @param children vector of geometry object for which index should be build
 * @return constructed index
 */
template <int DIMS>
std::unique_ptr<SpatialIndexNode<DIMS>> buildSpatialIndex(const std::vector< shared_ptr<Translation<DIMS>> >& children);

PLASK_API_EXTERN_TEMPLATE_STRUCT(SpatialIndexNode<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(SpatialIndexNode<3>)

extern template PLASK_API std::unique_ptr<SpatialIndexNode<2>> buildSpatialIndex(const std::vector< shared_ptr<Translation<2>> >& children);
extern template PLASK_API std::unique_ptr<SpatialIndexNode<3>> buildSpatialIndex(const std::vector< shared_ptr<Translation<3>> >& children);

}   // plask

#endif // PLASK__GEOMETRY_SPATIAL_INDEX_H
