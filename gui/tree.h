#ifndef PLASK_GUI_TREE_H
#define PLASK_GUI_TREE_H

#include <QAbstractItemModel>
#include <QPixmap>
#include <plask/geometry/element.h>
#include <plask/memory.h>

QT_BEGIN_NAMESPACE
class QAbstractItemModel;
class QObject;
class QModelIndex;
class QPixmap;
QT_END_NAMESPACE

/**
 * Geometry tree item. Holds (weak) geometry element.
 */
class GeometryTreeItem {

protected:

    QList<GeometryTreeItem*> childItems;

    /// Cache for miniature
    QPixmap miniature;

    //here can cache miniature

    /**
     * True only if this item was initialized (its children are on childItems list, etc.).
     */
    bool initialized;

    /**
     * Ensure that this item is initialized (initialize it if its not).
     */
    void ensureInitialized();

    /**
     * Construct and add to childItems children items for an given element.
     * @param elem element for which children should be constructed, typically (but not always) same as element
     */
    virtual void constructChildrenItems(const plask::shared_ptr<plask::GeometryElement>& elem);

public:

    /**
     * Index of this in parents item childItems. 0 for root.
     */
    std::size_t inParentIndex;

    /**
     * Parent of this in tree.
     */
    GeometryTreeItem* parentItem;

    /**
     * Wrapped geometry element.
     */
    plask::weak_ptr<plask::GeometryElement> element;

    /**
     * Get geometry element from parent item.
     * @return geometry element from parent item or plask::shared_ptr<plask::GeometryElement>()
     *  if can't get it (for example in case of root or parent doesn't wrap existing geometry element object).
     */
    plask::shared_ptr<plask::GeometryElement> parent();

    /**
     * Construct item using parent item and index in it.
     * @param parentItem parent item which wrap existing geometry element
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index);

    /**
     * Construct item.
     * @param parentItem parent item
     * @param element wrapped element
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<plask::GeometryElement>& element, std::size_t index);

    /**
     * Construct root item (with parentItem = nullptr).
     * @param rootElements children of roots element (showing in tree as roots)
     */
    GeometryTreeItem(const std::vector< plask::shared_ptr<plask::GeometryElement> >& rootElements);

    ///Delete children items.
    ~GeometryTreeItem();

    /**
     * Get child item with given @p index.
     * @param index index of child item
     * @return child item or nullptr in case of invalid index
     */
    GeometryTreeItem* child(std::size_t index);

    /**
     * Get number of children.
     * @return number of children
     */
    std::size_t childCount() { ensureInitialized(); return childItems.size(); }

    ///@return 1
    std::size_t columnCount() const { return 1; }

    ///@return inParentIndex
    std::size_t indexInParent() const;

    /**
     * Get text representation of element wrapped by this.
     * @param element element wrapped by this
     * @return text representation of an @p element
     */
    virtual QString elementText(plask::GeometryElement& element) const;

    /**
     * @return string returned by elementText or empty QVariant if this wraps non-existing element
     */
    QVariant data(int column) const;

    const QPixmap& icon() { ensureInitialized(); return miniature; }
};

/**
 * Wrap translation and child of this translation inside container
 * (this two elements are represented as one item in tree).
 */
struct InContainerTreeItem: public GeometryTreeItem {

    /**
     * @param parentItem parent item, must wrap plask::Translation<2> or plask::Translation<3>
     * @param index (future) index of this in parent childItems
     */
    InContainerTreeItem(GeometryTreeItem* parentItem, std::size_t index)
        : GeometryTreeItem(parentItem, index) {}

    virtual void constructChildrenItems(const plask::shared_ptr<plask::GeometryElement>& elem);

    virtual QString elementText(plask::GeometryElement& element) const;
};


class Document;

/**
 * Implementation of QAbstractItemModel which holds and use GeometryTreeItem.
 */
class GeometryTreeModel : public QAbstractItemModel {

    Q_OBJECT

    ///Root of tree, not wraps real geometry element but its children do that.
    GeometryTreeItem *rootItem;

public:

    /**
     * Refresh all tree content.
     * @param document document from which new content will be read
     */
    void refresh(Document& document);

    /**
     * @param document document from which tree content will be read
     * @param parent
     */
    GeometryTreeModel(Document& document, QObject *parent = 0);

    ///Delete rootItem.
    ~GeometryTreeModel();

    // ---------- implementation of QAbstractItemModel methods: --------

    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;

    QModelIndex parent(const QModelIndex &index) const;

    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant data(const QModelIndex &index, int role) const;

    Qt::ItemFlags flags(const QModelIndex &index) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
 };

#endif // PLASK_GUI_TREE_H
