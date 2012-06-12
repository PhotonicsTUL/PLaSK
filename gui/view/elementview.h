#ifndef PLASK_GUI_VIEW_ELEMENTVIEW_H
#define PLASK_GUI_VIEW_ELEMENTVIEW_H

#include <QAbstractItemView>
#include <QFont>
#include <QItemSelection>
#include <QItemSelectionModel>
#include <QModelIndex>
#include <QRect>
#include <QSize>
#include <QPoint>
#include <QWidget>

#include "../tree.h"

QT_BEGIN_NAMESPACE
class QRubberBand;
QT_END_NAMESPACE

/**
 * Geometry element viewer.
 */
class ElementViewer: public QAbstractItemView
{
    Q_OBJECT

public:
    ElementViewer(QWidget *parent = 0);

    QRect visualRect(const QModelIndex &index) const;
    void scrollTo(const QModelIndex &index, ScrollHint hint = EnsureVisible);
    QModelIndex indexAt(const QPoint &point) const;

    GeometryTreeModel* model() const { return static_cast<GeometryTreeModel*>(QAbstractItemView::model()); }
    void setModel (GeometryTreeModel* model) { QAbstractItemView::setModel(model); }

    plask::shared_ptr<ElementWrapper> getElementWrapper() const;
    plask::shared_ptr<plask::GeometryElementD<2> > getElement() const {
        plask::shared_ptr<ElementWrapper> ew = getElementWrapper();
        return ew ? plask::static_pointer_cast<plask::GeometryElementD<2>>(ew->wrappedElement) : plask::shared_ptr<plask::GeometryElementD<2> >();
    }

    plask::Box2d getBoundingBox() const {
        plask::shared_ptr<plask::GeometryElementD<2> > e = getElement();
        return e ? e->getBoundingBox() : plask::Box2d(0.0, 0.0, 0.0, 0.0);
    }

protected slots:
    void dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
    void rowsInserted(const QModelIndex &parent, int start, int end);
    void rowsAboutToBeRemoved(const QModelIndex &parent, int start, int end);

protected:
    bool edit(const QModelIndex &index, EditTrigger trigger, QEvent *event);
    QModelIndex moveCursor(QAbstractItemView::CursorAction cursorAction,
                           Qt::KeyboardModifiers modifiers);

    int horizontalOffset() const;
    int verticalOffset() const;

    bool isIndexHidden(const QModelIndex &index) const;

    void setSelection(const QRect&, QItemSelectionModel::SelectionFlags command);

    void mousePressEvent(QMouseEvent *event);

    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);
    void scrollContentsBy(int dx, int dy);

    QRegion visualRegionForSelection(const QItemSelection &selection) const;

private:
    QRect itemRect(const QModelIndex &item) const;
    QRegion itemRegion(const QModelIndex &index) const;
    int rows(const QModelIndex &index = QModelIndex()) const;
    void updateGeometries();

    int margin;
    int pieSize;
    int validItems;
    double totalValue;
    QPoint origin;
    QRubberBand *rubberBand;

    QPointF model_center;
    QPointF zoom;

    QTransform getTransformMatrix();

    void updateViewSize() {
        plask::Vec<2, double> s = getBoundingBox().size();
        viewport()->resize(s.c0 * zoom.x(), s.c1 * zoom.y());
    }
};

#endif // PLASK_GUI_VIEW_ELEMENTVIEW_H
