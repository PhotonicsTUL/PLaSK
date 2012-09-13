#include "path.h"

#include <plask/config.h>

namespace plask {

void PathHints::addHint(const Hint& hint) {
    addHint(hint.first, hint.second);
}

void PathHints::addHint(weak_ptr<GeometryObject> container, weak_ptr<GeometryObject> child) {
    hintFor[container].insert(child);
}

void PathHints::addAllHintsFromPath(const std::vector< shared_ptr<const GeometryObject> >& pathObjects) {
    int possibleContainers_size = pathObjects.size() - 1;
    for (int i = 0; i < possibleContainers_size; ++i)
        if (pathObjects[i]->isContainer())
            addHint(const_pointer_cast<GeometryObject>(pathObjects[i]), const_pointer_cast<GeometryObject>(pathObjects[i+1]));
}

void PathHints::addAllHintsFromPath(const Path& path) {
    addAllHintsFromPath(path.objects);
}

void PathHints::addAllHintsFromSubtree(const GeometryObject::Subtree &subtree) {
    if (subtree.object->isContainer()) {
        for (auto& c: subtree.children)
            addHint(const_pointer_cast<GeometryObject>(subtree.object), const_pointer_cast<GeometryObject>(c.object));
    }
    for (auto& c: subtree.children)
        addAllHintsFromPath(c);
}

std::set<shared_ptr<GeometryObject>> PathHints::getChildren(shared_ptr<const GeometryObject> container) {
    std::set<shared_ptr<GeometryObject>> result;
    auto e = hintFor.find(const_pointer_cast<GeometryObject>(container));
    if (e == hintFor.end()) return result;
    if (e->first.expired()) {   // container was deleted, new container is under same address as was old one
        hintFor.erase(e);
        return result;
    }
    for (auto weak_child_iter = e->second.begin(); weak_child_iter != e->second.end(); ) {
        shared_ptr<GeometryObject> child = weak_child_iter->lock();
        if (!child)        // child was deleted
            e->second.erase(weak_child_iter++);
        else {
            result.insert(child);
            ++weak_child_iter;
        }
    }
    if (e->second.empty()) hintFor.erase(e);    // we remove all constraints
    return result;
}

std::set<shared_ptr<GeometryObject>> PathHints::getChildren(shared_ptr<const GeometryObject> container) const {
    std::set<shared_ptr<GeometryObject>> result;
    auto e = hintFor.find(const_pointer_cast<GeometryObject>(container));
    if (e == hintFor.end() || e->first.expired()) return result;
    for (auto child_weak: e->second) {
        shared_ptr<GeometryObject> child = child_weak.lock();
        if (child) result.insert(child);
    }
    return result;
}

void PathHints::cleanDeleted() {
    for(auto i = hintFor.begin(); i != hintFor.end(); )
        if (i->first.expired())
            hintFor.erase(i++);
        else {
            for (auto weak_child_iter = i->second.begin(); weak_child_iter != i->second.end(); ) {
                if (weak_child_iter->expired())
                    i->second.erase(weak_child_iter);
                else
                    ++weak_child_iter;
            }
            if (i->second.empty())
                hintFor.erase(i++);
            else
                ++i;
        }
}

//----------------- Path ------------------------------------------

bool Path::completeToFirst(const GeometryObject& newFirst, const PathHints* hints) {
    GeometryObject::Subtree path = newFirst.getPathsTo(*objects.front(), hints);
    if (path.empty()) return false;
    push_front(path.toLinearPath());
    return true;
}

bool Path::completeFromLast(const GeometryObject& newLast, const PathHints* hints) {
    GeometryObject::Subtree path = objects.back()->getPathsTo(newLast, hints);
    if (path.empty()) return false;
    push_back(path.toLinearPath());
    return true;
}

void Path::push_front(const std::vector< shared_ptr<const GeometryObject> >& toAdd) {
    if (toAdd.empty()) return;
    if (objects.empty()) {
        objects = toAdd;
    } else {
        if (toAdd.back() == objects.front())   //last to add is already first on list?
            objects.insert(objects.begin(), toAdd.begin(), toAdd.end()-1);
        else
            objects.insert(objects.begin(), toAdd.begin(), toAdd.end());
    }
}

void Path::push_back(const std::vector< shared_ptr<const GeometryObject> >& toAdd) {
    if (toAdd.empty()) return;
    if (objects.empty()) {
        objects = toAdd;
    } else {
        if (toAdd.front() == objects.back())   //first to add is already as last on list?
            objects.insert(objects.end(), toAdd.begin()+1, toAdd.end());
        else
            objects.insert(objects.end(), toAdd.begin(), toAdd.end());
    }
}

Path& Path::append(const std::vector< shared_ptr<const GeometryObject> >& path, const PathHints* hints) {
    if (path.empty()) return *this;
    if (objects.empty())
        objects = path;
    else {
        if (completeToFirst(*path.back(), hints)) {
            push_front(path);
        } else
        if (completeFromLast(*path.front(), hints)) {
            push_back(path);
        } else
            throw Exception("Cannott connect paths.");
    }
    return *this;
}

Path& Path::append(const GeometryObject::Subtree& path, const PathHints* hints) {
    return append(path.toLinearPath(), hints);
}

Path& Path::append(const Path& path, const PathHints* hints) {
    return append(path.objects, hints);
}

Path& Path::append(const PathHints::Hint& hint, const PathHints* hints) {
    return append(std::vector< shared_ptr<const GeometryObject> > { hint.first, hint.second }, hints);
}

Path& Path::append(const GeometryObject& object, const PathHints* hints) {
    return append( std::vector< shared_ptr<const GeometryObject> > { object.shared_from_this() }, hints);
}

Path& Path::append(shared_ptr<const GeometryObject> object, const PathHints* hints) {
    return append( std::vector< shared_ptr<const GeometryObject> > { object }, hints);
}

PathHints Path::getPathHints() const {
    PathHints result;
    result.addAllHintsFromPath(*this);
    return result;
}

}   //namespace plask
