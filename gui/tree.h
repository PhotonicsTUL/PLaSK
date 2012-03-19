#ifndef PLASK_GUI_TREE_H
#define PLASK_GUI_TREE_H

#include <QAbstractItemModel>
#include <plask/geometry/element.h>
#include <plask/memory.h>

QT_BEGIN_NAMESPACE
class QAbstractItemModel;
class QObject;
class QModelIndex;
QT_END_NAMESPACE

/**
 * Geometry tree item. Holds (weak) geometry element.
 */
class GeometryTreeItem {

protected:

    QList<GeometryTreeItem*> childItems;

    //here can cache miniature

    bool initialized;

    void ensureInitialized();

    /**
     * Construct and add to childItems children items for an given element.
     * @param elem element for which children should be constructed, typically (but not always) same as element
     */
    virtual void constructChildrenItems(const plask::shared_ptr<plask::GeometryElement>& elem);

public:

    std::size_t inParentIndex;

    GeometryTreeItem* parentItem;

    plask::weak_ptr<plask::GeometryElement> element;

    plask::shared_ptr<plask::GeometryElement> parent();

    GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index);

    GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<plask::GeometryElement>& element, std::size_t index);

    //create root with parentItem = nullptr
    GeometryTreeItem(const std::vector< plask::shared_ptr<plask::GeometryElement> >& rootElements);

    ~GeometryTreeItem();

    GeometryTreeItem* child(std::size_t index);

    std::size_t childCount() { ensureInitialized(); return childItems.size(); }

    std::size_t columnCount() const { return 1; }

    std::size_t indexInParent() const;

    virtual QString elementText(plask::GeometryElement& element) const;

    QVariant data(int column) const;
};


struct InContainerTreeItem: public GeometryTreeItem {

    InContainerTreeItem(GeometryTreeItem* parentItem, std::size_t index)
        : GeometryTreeItem(parentItem, index) {}

    virtual void constructChildrenItems(const plask::shared_ptr<plask::GeometryElement>& elem);

    virtual QString elementText(plask::GeometryElement& element) const;
};


class Document;

class GeometryTreeModel : public QAbstractItemModel {

    Q_OBJECT

    GeometryTreeItem *rootItem;

public:

    void refresh(Document& document);

    GeometryTreeModel(Document& document, QObject *parent = 0);

    ~GeometryTreeModel();

    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;

    QModelIndex parent(const QModelIndex &index) const;

    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant data(const QModelIndex &index, int role) const;

    Qt::ItemFlags flags(const QModelIndex &index) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
 };

#endif // PLASK_GUI_TREE_H
