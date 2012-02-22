#include "path.h"

namespace plask {

void PathHints::addHint(const Hint& hint) {
    addHint(hint.first, hint.second);
}

void PathHints::addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child) {
    hintFor[container] = child;
}

shared_ptr<GeometryElement> PathHints::getChild(shared_ptr<const GeometryElement> container) {
    auto e = hintFor.find(const_pointer_cast<GeometryElement>(container));
    if (e == hintFor.end()) return shared_ptr<GeometryElement>();
    shared_ptr<GeometryElement> result = e->second.lock();
    if (!result || e->first.expired()) {        //child or container was deleted (in second case, new container is under same address as was old one)
        hintFor.erase(const_pointer_cast<GeometryElement>(container));
        return shared_ptr<GeometryElement>();
    }
    return result;
}

shared_ptr<GeometryElement> PathHints::getChild(shared_ptr<const GeometryElement> container) const {
    auto e = hintFor.find(const_pointer_cast<GeometryElement>(container));
    return (e == hintFor.end() || e->first.expired()) ? shared_ptr<GeometryElement>() : e->second.lock();
}

shared_ptr<GeometryElement> PathHints::getChild(const Hint& hint) {
    return (hint.first.expired()) ? shared_ptr<GeometryElement>() : hint.second.lock();
}

shared_ptr<GeometryElement> PathHints::getContainer(const Hint& hint) {
    return (hint.first.expired()) ? shared_ptr<GeometryElement>() : hint.first.lock();
}


void PathHints::cleanDeleted() {
    for(auto i = hintFor.begin(); i != hintFor.end(); )
        if (i->first.expired() || i->second.expired())
            hintFor.erase(i++);
        else
            ++i;
}

}   //namespace plask
