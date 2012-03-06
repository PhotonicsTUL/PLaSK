#ifndef PLASK__GEOMETRY_TRANSFORM_H
#define PLASK__GEOMETRY_TRANSFORM_H

#include "element.h"

namespace plask {

/**
 * Represent geometry element equal to its child translated by vector.
 */
template <int dim>
struct Translation: public GeometryElementTransform<dim> {

    typedef typename GeometryElementTransform<dim>::ChildType ChildType;

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::DVec DVec;
    
    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::Rect Rect;
    
    using GeometryElementTransform<dim>::getChild;

    /**
     * Translation vector.
     */
    DVec translation;

    //Translation(const Translation<dim>& translation) = default;

    /**
     * @param child child geometry element, element to translate
     * @param translation translation
     */
    explicit Translation(shared_ptr< GeometryElementD<dim> > child = shared_ptr< GeometryElementD<dim> >(), const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(child), translation(translation) {}

    explicit Translation(GeometryElementD<dim>& child, const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(child), translation(translation) {}

    virtual Rect getBoundingBox() const {
        return getChild()->getBoundingBox().translated(translation);
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return getChild()->getMaterial(p-translation);
    }

    virtual bool inside(const DVec& p) const {
        return getChild()->inside(p-translation);
    }

    virtual bool intersect(const Rect& area) const {
        return getChild()->intersect(area.translated(-translation));
    }

    /*virtual void getLeafsInfoToVec(std::vector< std::tuple<shared_ptr<const GeometryElement>, Rect, DVec> >& dest, const PathHints* path = 0) const {
        const std::size_t old_size = dest.size();
        getChild()->getLeafsInfoToVec(dest, path);
        for (auto i = dest.begin() + old_size; i != dest.end(); ++i) {
            std::get<1>(*i).translate(translation);
            std::get<2>(*i) += translation;
        }
    }*/

    virtual void getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path = 0) const {
        std::vector<Rect> result = getChild()->getLeafsBoundingBoxes(path);
        dest.reserve(dest.size() + result.size());
        for (Rect& r: result) dest.push_back(r.translated(translation));
    }
    
    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > result = getChild()->getLeafsWithTranslations();
        for (std::tuple<shared_ptr<const GeometryElement>, DVec>& r: result) std::get<1>(r) += translation;
        return result;
    }

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Translation<dim>> copyShallow() const {
         return shared_ptr<Translation<dim>>(new Translation<dim>(getChild(), translation));
    }

    virtual shared_ptr<GeometryElementTransform<dim>> shallowCopy() const {
        return copyShallow();
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        if (changer.apply(result, translation) || !this->hasChild()) return result;
        Vec<3, double> returned_translation(0.0, 0.0, 0.0);
        shared_ptr<const GeometryElement> new_child = this->getChild()->changedVersion(changer, &returned_translation);
        Vec<dim, double> translation_we_will_do = vec<dim, double>(returned_translation);
        if (new_child == getChild() && translation_we_will_do == Primitive<dim>::ZERO_VEC) return result;
        if (translation)    //we will change translation (partially if dim==2) internaly, so we recommend no extra translation
            *translation = returned_translation - vec<3, double>(translation_we_will_do); //still we can recommend translation in third direction
        return shared_ptr<GeometryElement>(
            new Translation<dim>(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)),
            this->translation + translation_we_will_do) );
    }

    /**
     * Get shallow, moved copy of this.
     * @param new_translation translation vector of copy
     */
    shared_ptr<Translation<dim>> copyShallow(const DVec& new_translation) const {
        return shared_ptr<Translation<dim>>(new Translation<dim>(getChild(), new_translation));
    }

};

}       // namespace plask

#endif // PLASK__GEOMETRY_TRANSFORM_H
