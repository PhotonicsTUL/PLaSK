#include "stack.h"

namespace plask {

bool HorizontalStack::allChildrenHaveSameHeights() const {
    if (children.size() < 2) return true;
    double height = children.front()->getBoundingBoxSize().up;
    for (std::size_t i = 1; i < children.size(); ++i)
        if (height != children[i]->getBoundingBoxSize().up)
            return false;
    return true;
}

PathHints::Hint HorizontalStack::addUnsafe(const shared_ptr<ChildType>& el) {
    double el_translation, next_height;
    auto elBB = el->getBoundingBox();
    calcHeight(elBB, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el, vec(el_translation, -elBB.lower.up));
    connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    this->fireChildrenChanged();
    return PathHints::Hint(shared_from_this(), trans_geom);
}

#define baseH_attr "from"
#define repeat_attr "repeat"

shared_ptr<GeometryElement> read_StackContainer2d(GeometryReader& reader) {
    const double baseH = reader.source.getAttribute(baseH_attr, 0.0);
    std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> default_aligner(
          align::fromStr<align::DIRECTION_TRAN>(reader.source.getAttribute<std::string>(reader.getAxisTranName(), "c")));

    shared_ptr< StackContainer<2> > result(
                    reader.source.hasAttribute(repeat_attr) ?
                    new MultiStackContainer<2>(reader.source.getAttribute(repeat_attr, 1), baseH) :
                    new StackContainer<2>(baseH)
                );
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<StackContainer<2>>(reader,
            [&]() {
                boost::optional<std::string> aligner_str = reader.source.getAttribute(reader.getAxisTranName());
                if (aligner_str) {
                   std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> aligner(align::fromStr<align::DIRECTION_TRAN>(*aligner_str));
                   return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *aligner);
                } else {
                   return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *default_aligner);
                }
            },
            [&](const shared_ptr<typename StackContainer<2>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

shared_ptr<GeometryElement> read_StackContainer3d(GeometryReader& reader) {
    const double baseH = reader.source.getAttribute(baseH_attr, 0.0);
    //TODO default aligner (see above)
    shared_ptr< StackContainer<3> > result(
                    reader.source.hasAttribute(repeat_attr) ?
                    new MultiStackContainer<3>(reader.source.getAttribute(repeat_attr, 1), baseH) :
                    new StackContainer<3>(baseH)
                );
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<StackContainer<3>>(reader,
            [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<3>::ChildType >(),
                                          align::fromStr<align::DIRECTION_LON, align::DIRECTION_TRAN>(
                                              reader.source.getAttribute<std::string>(reader.getAxisLonName(), "c"),
                                              reader.source.getAttribute<std::string>(reader.getAxisTranName(), "c")
                                          ));
            },
            [&](const shared_ptr<typename StackContainer<3>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

static GeometryReader::RegisterElementReader stack2d_reader("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_StackContainer2d);
static GeometryReader::RegisterElementReader stack3d_reader("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_StackContainer3d);

}   // namespace plask
