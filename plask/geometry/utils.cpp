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
#include "utils.hpp"

namespace plask {

template <int dims>
void BoundingBoxCache<dims>::connect() {
    if (this->object)
         connection_with_object = this->object->changed.connect(boost::bind(&BoundingBoxCache<dims>::onObjectChanged, this, _1), boost::signals2::at_front);
}

template <int dims>
const typename BoundingBoxCache<dims>::BoundingBoxT &BoundingBoxCache<dims>::operator()() {
    if (!isFresh) {
        //if (!object) return BoundingBoxT::invalidInstance();
        if (!object) throw Exception("BoundingBoxCache is not initialized or object was deleted, so can't get bounding box");
        boundingBox = object->getBoundingBox();
        isFresh = true;
    }
    return boundingBox;
}

template <int dims>
void BoundingBoxCache<dims>::onObjectChanged(const GeometryObject::Event &evt) {
    //if (evt.isResize())
    //    boundingBox = static_cast<GeometryObjectD<dims>&>(evt.source())->getBoundingBox();
    if (evt.isResize()) isFresh = false;
    if (evt.isDelete()) {
        object = nullptr;
        isFresh = false;
    }
}

template <int dims>
void BoundingBoxCache<dims>::setObject(GeometryObjectD<dims> *object) {
    if (this->object == object) return;
    connection_with_object.disconnect();
    this->object = object;
    isFresh = false;
    connect();
}


template struct PLASK_API BoundingBoxCache<2>;
template struct PLASK_API BoundingBoxCache<3>;

}   // namespace plask
