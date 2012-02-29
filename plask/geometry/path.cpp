#include "path.h"

#include <plask/config.h>

namespace plask {

void PathHints::addHint(const Hint& hint) {
    addHint(hint.first, hint.second);
}

void PathHints::addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child) {
    hintFor[container].insert(child);
}

std::set<shared_ptr<GeometryElement>> PathHints::getChildren(shared_ptr<const GeometryElement> container) {
    std::set<shared_ptr<GeometryElement>> result;
    auto e = hintFor.find(const_pointer_cast<GeometryElement>(container));
    if (e == hintFor.end()) return result;
    if (e->first.expired()) {   //container was deleted, new container is under same address as was old one
        hintFor.erase(e);
        return result;
    }
    for (auto weak_child_iter = e->second.begin(); weak_child_iter != e->second.end(); ) {
        shared_ptr<GeometryElement> child = weak_child_iter->lock();
        if (!child)        //child was deleted
            e->second.erase(weak_child_iter++);
        else {
            result.insert(child);
            ++weak_child_iter;
        }
    }
    if (e->second.empty()) hintFor.erase(e);    //we remove all constraints
    return result;
}

std::set<shared_ptr<GeometryElement>> PathHints::getChildren(shared_ptr<const GeometryElement> container) const {
    std::set<shared_ptr<GeometryElement>> result;
    auto e = hintFor.find(const_pointer_cast<GeometryElement>(container));
    if (e == hintFor.end() || e->first.expired()) return result;
    for (auto child_weak: e->second) {
        shared_ptr<GeometryElement> child = child_weak.lock();
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

bool Path::complateToFirst(const GeometryElement& newFirst, const PathHints* hints) {
    GeometryElement::Subtree path = newFirst.findPathsTo(*elements.front(), hints);
    if (path.empty()) return false;
    push_front(path.toLinearPath());
    return true;
}

bool Path::complateFromLast(const GeometryElement& newLast, const PathHints* hints) {
    GeometryElement::Subtree path = elements.back()->findPathsTo(newLast, hints);
    if (path.empty()) return false;
    push_back(path.toLinearPath());
    return true;
}

void Path::push_front(const std::vector< shared_ptr<const GeometryElement> >& toAdd) {
    if (toAdd.empty()) return;
    if (elements.empty()) {
        elements = toAdd;
    } else {
        if (toAdd.back() == elements.front())   //last to add is already first on list?
            elements.insert(elements.begin(), toAdd.begin(), toAdd.end()-1);
        else
            elements.insert(elements.begin(), toAdd.begin(), toAdd.end());
    }
}

void Path::push_back(const std::vector< shared_ptr<const GeometryElement> >& toAdd) {
    if (toAdd.empty()) return;
    if (elements.empty()) {
        elements = toAdd;
    } else {
        if (toAdd.front() == elements.back())   //first to add is already as last on list?
            elements.insert(elements.end(), toAdd.begin()+1, toAdd.end());
        else
            elements.insert(elements.end(), toAdd.begin(), toAdd.end());
    }
}

Path& Path::append(const std::vector< shared_ptr<const GeometryElement> >& path, const PathHints* hints) {
    if (path.empty()) return *this;
    if (elements.empty())
        elements = path;
    else {
        if (complateToFirst(*path.back(), hints)) {
            push_front(path);
        } else
        if (complateFromLast(*path.front(), hints)) {
            push_back(path);
        } else
            throw Exception("Can't connect paths.");
    }
    return *this;
}

Path& Path::append(const GeometryElement::Subtree& paths, const PathHints* hints) {
    return append(paths.toLinearPath(), hints);
}

Path& Path::append(const Path& path, const PathHints* hints) {
    return append(path.elements, hints);
}

Path& Path::append(const PathHints::Hint& hint, const PathHints* hints) {
    return append(std::vector< shared_ptr<const GeometryElement> > { hint.first, hint.second }, hints);
}

Path& Path::append(const GeometryElement& element, const PathHints* hints) {
    return append( std::vector< shared_ptr<const GeometryElement> > { element.shared_from_this() }, hints);
}

}   //namespace plask
