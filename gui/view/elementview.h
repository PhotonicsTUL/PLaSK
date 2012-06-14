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

    /**
       @return the position of the item in viewport coordinates.
    */
    QRect visualRect(const QModelIndex &index) const;

    void scrollTo(const QModelIndex &index, ScrollHint hint = EnsureVisible);

    /**
        Returns the item that covers the coordinate given in the view.
    */
    QModelIndex indexAt(const QPoint &point) const;

    GeometryTreeModel* model() const { return static_cast<GeometryTreeModel*>(QAbstractItemView::model()); }
    void setModel (GeometryTreeModel* model) { QAbstractItemView::setModel(model); }

    plask::shared_ptr<ElementWrapper> getElementWrapper() const;
    plask::shared_ptr<plask::GeometryElementD<2> > getElement() const {
        plask::shared_ptr<ElementWrapper> ew = getElementWrapper();
        return ew ? plask::static_pointer_cast<plask::GeometryElementD<2>>(ew->wrappedElement) : plask::shared_ptr<plask::GeometryElementD<2> >();
    }

    /**
     * Get model bounding box in model coordinates.
     */
    plask::Box2d getBoundingBox() const {
        plask::shared_ptr<plask::GeometryElementD<2> > e = getElement();
        return e ? e->getRealBoundingBox() : plask::Box2d(0.0, 0.0, 0.0, 0.0);
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

    /**
      Find the indices corresponding to the extent of the selection.
    */
    void setSelection(const QRect&, QItemSelectionModel::SelectionFlags command);

    void mousePressEvent(QMouseEvent *event);

    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);
    void scrollContentsBy(int dx, int dy);

    /**
      @return a region corresponding to the selection in viewport coordinates
    */
    QRegion visualRegionForSelection(const QItemSelection &selection) const;

private:
    /**
        Returns the rectangle of the item at position \a index in the
        model. The rectangle is in contents coordinates.
    */
    QRectF itemRect(const QModelIndex &item) const;
    QRegion itemRegion(const QModelIndex &index) const;
    int rows(const QModelIndex &index = QModelIndex()) const;
    void updateGeometries();

    int margin;
    QPoint origin;
    QRubberBand *rubberBand;

    QPointF model_center;
    QPointF zoom;

    /**
     * Get transform matrix which can transform model coordinates to view one.
     */
    QTransform getTransformMatrix() const;

public slots:
    void zoomIn() {
        if (zoom.x() < 1000.0 && zoom.y() < 1000.0) {
            zoom.rx() *= 1.2;
            zoom.ry() *= 1.2;
            updateGeometries();
            viewport()->update();
        }
    }

    void zoomOut() {
        if (zoom.x() > 0.001 && zoom.y() > 0.001) {
            zoom.rx() /= 1.2;
            zoom.ry() /= 1.2;
            updateGeometries();
            viewport()->update();
        }
    }
};

#endif // PLASK_GUI_VIEW_ELEMENTVIEW_H
