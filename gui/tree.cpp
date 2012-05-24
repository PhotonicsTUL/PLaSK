#include "tree.h"

#include "document.h"
#include "geomwrapper/register.h"
#include "modelext/text.h"

void GeometryTreeItem::ensureInitialized() {
    if (!childrenInitialized) {
        constructChildrenItems();
        childrenInitialized = true;
    }
    if (!miniatureInitialized) {
        miniature = element->getMiniature(50, 50);
        miniatureInitialized = true;
    }
}

void GeometryTreeItem::appendChildrenItemsHelper(const plask::shared_ptr<plask::GeometryElement>& elem, bool reverse) {
    //TODO reverse support
    std::size_t chCount = elem->getRealChildrenCount();
    if (elem->isContainer()) {
        for (int i = 0; i < chCount; ++i)
            childItems.emplace_back(new InContainerTreeItem(this, ext(elem->getRealChildAt(i)), i));
    } else {
        for (int i = 0; i < chCount; ++i)   //should be 0 or 1 child here
            childItems.emplace_back(new GeometryTreeItem(this, ext(elem->getRealChildAt(i)), i));
    }
}

void GeometryTreeItem::appendChildrenItems() {
    if (auto e = getLowerWrappedElement()) {
        appendChildrenItemsHelper(e->wrappedElement);
    }
}

void GeometryTreeItem::constructChildrenItems() {
    childItems.clear();
    appendChildrenItems();
}

void GeometryTreeItem::deinitializeChildren()
{
    childItems.clear();
    childrenInitialized = false;
}

plask::shared_ptr<ElementWrapper> GeometryTreeItem::parent() {
    return parentItem ?
        parentItem->getLowerWrappedElement() :
        plask::shared_ptr<ElementWrapper>();
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index)
: model(parentItem->model), childrenInitialized(false), miniatureInitialized(false), parentItem(parentItem)/*, inParentIndex(index)*/ {
    if (auto parent_ptr = parent()) {
        auto child = ext(parent_ptr->wrappedElement->getRealChildAt(index));
        element = child;
        connectOnChanged(child);
        //constructChildrenItems(child);
    }
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<ElementWrapper> &element, std::size_t index)
: model(parentItem->model), childrenInitialized(false), miniatureInitialized(false), parentItem(parentItem), element(element)/*, inParentIndex(index)*/ {
    connectOnChanged(element);
    //constructChildrenItems(element);
}

GeometryTreeItem::GeometryTreeItem(const std::vector< plask::shared_ptr<plask::GeometryElement> >& rootElements, GeometryTreeModel* model)
: model(model), childrenInitialized(true), miniatureInitialized(true), parentItem(0)/*, inParentIndex(0)*/ {
    for (int i = 0; i < rootElements.size(); ++i)
        childItems.emplace_back(new GeometryTreeItem(this, ext(rootElements[i]), i));
}

GeometryTreeItem::~GeometryTreeItem() {
    disconnectOnChanged(element);
}

GeometryTreeItem * GeometryTreeItem::child(std::size_t index) {
    ensureInitialized();
    if (index > childItems.size()) return nullptr;
    return childItems[index].get();
}

std::size_t GeometryTreeItem::indexInParent() const {
    if (!parentItem) return 0;
    for (std::size_t i = 0; i < parentItem->childItems.size(); ++i)
        if (parentItem->childItems[i].get() == this) return i;
    //TODO
    return 0;

//    return std::find(parentItem->childItems.begin(), parentItem->childItems.end(),
//                     [&](std::unique_ptr<GeometryTreeItem>& p) { return p.get() == this; }) - parentItem->childItems.begin();

   // return parentItem->childItems.indexOf(const_cast<GeometryTreeItem*>(this));
   // return inParentIndex;
}

QString GeometryTreeItem::elementText(plask::shared_ptr<ElementWrapper> element) const {
    return element->toStr();
}

QVariant GeometryTreeItem::data(int column) {
    return elementText(element);
}

void GeometryTreeItem::fillPropertyBrowser(BrowserWithManagers& browser) {
    element->setupPropertiesBrowser(browser);
}

QModelIndex GeometryTreeItem::getIndex() {
    return model->createIndex(indexInParent(), 0, this);
}

void GeometryTreeItem::onChanged(const ElementWrapper::Event& evt) {
    auto index = getIndex();
    miniatureInitialized = false;
    if (evt.hasChangedChildrenList()) {
        //deinitializeChildren();   //TODO
    }
    emit model->dataChanged(index, index);  //TODO czy ten sygnał jest wystarczający jeśli lista dzieci się zmieniła?
}

void GeometryTreeItem::connectOnChanged(const plask::shared_ptr<ElementWrapper>& el) {
    if (el) el->changedConnectMethod(this, &GeometryTreeItem::onChanged);
}

void GeometryTreeItem::disconnectOnChanged(const plask::shared_ptr<ElementWrapper> &el) {
    if (el) el->changedDisconnectMethod(this, &GeometryTreeItem::onChanged);
}

bool GeometryTreeItem::removeRange(std::size_t begin_index, std::size_t end_index) {
    if (auto e = getLowerWrappedElement()) {
        if (e->wrappedElement->removeRange(begin_index, end_index)) {
            childItems.erase(childItems.begin() + begin_index, childItems.begin() + end_index);
            //deinitializeChildren();
            return true;
        }
    }
    return false;
}

// ---------- InContainerTreeItem -----------

/*void InContainerTreeItem::appendChildrenItems() {
    if (auto elem = element.lock()) {
        std::size_t chCount = elem->getRealChildrenCount();
        if (chCount == 0) return;
        appendChildrenItemsHelper(elem->getRealChildAt(0));
    }
}*/

QString InContainerTreeItem::elementText(plask::shared_ptr<ElementWrapper> element) const {
    if (element->wrappedElement->getRealChildrenCount() == 0) return element->toStr();
    QString result = ext(element->wrappedElement->getRealChildAt(0))->toStr();
    result += "\nat ";
    if (element->wrappedElement->getDimensionsCount() == 2) {
        result += toStr(static_cast<plask::Translation<2>&>(*element->wrappedElement).translation);
    } else
        result += toStr(static_cast<plask::Translation<3>&>(*element->wrappedElement).translation);
    return result;
}

void InContainerTreeItem::fillPropertyBrowser(BrowserWithManagers& browser) {
    auto p = parent();  //should be a container
    if (p) {
        p->setupPropertiesBrowserForChild(indexInParent(), browser);
    } else {
        if (element->wrappedElement->getRealChildrenCount() == 0) return;
        ext(element->wrappedElement->getRealChildAt(0))->setupPropertiesBrowser(browser);
    }
}

// ----------- GeometryTreeModel ------------

GeometryTreeModel::GeometryTreeModel(Document& document, QObject *parent)
    : QAbstractItemModel(parent), rootItem(0) {
    refresh(document);
}

GeometryTreeModel::~GeometryTreeModel() {
    delete rootItem;
}

void GeometryTreeModel::refresh(Document& document) {
    delete rootItem;
    rootItem = new GeometryTreeItem(document.manager.roots, this);
    reset();
}

QModelIndex GeometryTreeModel::index(int row, int column, const QModelIndex &parent) const {
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    GeometryTreeItem* parentItem = parent.isValid() ?
                static_cast<GeometryTreeItem*>(parent.internalPointer()) :
                rootItem;

    GeometryTreeItem *childItem = parentItem->child(row);
    return childItem ?
        createIndex(row, column, childItem) :
        QModelIndex();
}

QModelIndex GeometryTreeModel::parent(const QModelIndex &index) const {
    if (!index.isValid())
        return QModelIndex();

    GeometryTreeItem *childItem = static_cast<GeometryTreeItem*>(index.internalPointer());
    GeometryTreeItem *parentItem = childItem->parentItem;

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->indexInParent(), 0, parentItem);
}

int GeometryTreeModel::rowCount(const QModelIndex &parent) const {
    if (parent.column() > 0)    //TODO
        return 0;
    return toItem(parent)->childCount();
}

int GeometryTreeModel::columnCount(const QModelIndex &parent) const {
    return toItem(parent)->columnCount();
}

QVariant GeometryTreeModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid())
        return QVariant();

    GeometryTreeItem* item = static_cast<GeometryTreeItem*>(index.internalPointer());

    if (role == Qt::DecorationRole)
        return item->icon();

    if (role != Qt::DisplayRole)
        return QVariant();

    return item->data(index.column());
}

Qt::ItemFlags GeometryTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable; //| Qt::ItemIsEditable;
}

QVariant GeometryTreeModel::headerData(int section, Qt::Orientation orientation, int role) const {

    if (role != Qt::DisplayRole)
        return QVariant();

    return "Description";
}

bool GeometryTreeModel::removeRows(int position, int rows, const QModelIndex &parent) {
    beginRemoveRows(parent, position, position + rows - 1);
    bool result = toItem(parent)->remove(position, rows);
    endRemoveRows();
    return result;
}
