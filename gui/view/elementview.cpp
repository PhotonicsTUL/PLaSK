#include "elementview.h"

#include <math.h>
#include <QtGui>

#include "elementview.h"

#include "../modelext/converter.h"

ElementViewer::ElementViewer(QWidget *parent)
    : QAbstractItemView(parent), zoom(10.0, 10.0)
{
    horizontalScrollBar()->setRange(0, 0);
    verticalScrollBar()->setRange(0, 0);

    margin = 8;
    rubberBand = 0;
}

void ElementViewer::dataChanged(const QModelIndex &topLeft,
                          const QModelIndex &bottomRight)
{
    QAbstractItemView::dataChanged(topLeft, bottomRight);
    updateGeometries(); //bounding-box could be changed
    viewport()->update();   //redraw
}

bool ElementViewer::edit(const QModelIndex &index, EditTrigger trigger, QEvent *event)
{
    /*if (index.column() == 0)
        return QAbstractItemView::edit(index, trigger, event);
    else*/
        return false;
}

QModelIndex ElementViewer::indexAt(const QPoint &point) const
{
    if (!rootIndex().isValid()) return QModelIndex();   //model not set

    plask::Vec<2, double> model_point = fromQt(getTransformMatrix().inverted().map(QPointF(point)));

    plask::shared_ptr<plask::GeometryElementD<2> > e = getElement();
    const std::size_t ch_count = e->getRealChildrenCount();
    for (std::size_t i = 0; i < ch_count; ++i) {
        plask::shared_ptr<plask::GeometryElementD<2> > ch = e->getRealChildAt(i)->asD<2>();
        if (ch && ch->getBoundingBox().inside(model_point))
            return model()->index(i, 0, rootIndex());
    }

    return QModelIndex();
}

plask::shared_ptr<ElementWrapper> ElementViewer::getElementWrapper() const
{
    QModelIndex toDrawIndex = rootIndex();
    if (!toDrawIndex.isValid()) return plask::shared_ptr<ElementWrapper>();    //we don't work on root
    return model()->toItem(toDrawIndex)->getLowerWrappedElement();
}

bool ElementViewer::isIndexHidden(const QModelIndex & /*index*/) const
{
    return false;
}

QRectF ElementViewer::itemRect(const QModelIndex &index) const
{
    if (index.isValid() && index.parent() == rootIndex()) {
        GeometryTreeItem* i = model()->toItem(index);
        plask::shared_ptr<plask::GeometryElement> e = i->element->wrappedElement;
        if (e->getDimensionsCount() == 2) {
            plask::Box2d b = plask::static_pointer_cast<plask::GeometryElementD<2> >(e)->getBoundingBox();
            return toQt(b);
        }
    }
    return QRectF();
}

int ElementViewer::horizontalOffset() const
{
    return horizontalScrollBar()->value();
}

void ElementViewer::mousePressEvent(QMouseEvent *event)
{
    QAbstractItemView::mousePressEvent(event);
    origin = event->pos();
    if (!rubberBand)
        rubberBand = new QRubberBand(QRubberBand::Rectangle, viewport());
    rubberBand->setGeometry(QRect(origin, QSize()));
    rubberBand->show();
}

void ElementViewer::mouseMoveEvent(QMouseEvent *event)
{
    if (rubberBand)
        rubberBand->setGeometry(QRect(origin, event->pos()).normalized());
    QAbstractItemView::mouseMoveEvent(event);
}

void ElementViewer::mouseReleaseEvent(QMouseEvent *event)
{
    QAbstractItemView::mouseReleaseEvent(event);
    if (rubberBand)
        rubberBand->hide();
    viewport()->update();
}

QModelIndex ElementViewer::moveCursor(QAbstractItemView::CursorAction cursorAction,
                                Qt::KeyboardModifiers /*modifiers*/)
{
    QModelIndex current = currentIndex();

    switch (cursorAction) {
        case MoveLeft:
        case MoveUp:
            if (current.row() > 0)
                current = model()->index(current.row() - 1, current.column(),
                                         rootIndex());
            else
                current = model()->index(0, current.column(), rootIndex());
            break;
        case MoveRight:
        case MoveDown:
            if (current.row() < rows(current) - 1)
                current = model()->index(current.row() + 1, current.column(),
                                         rootIndex());
            else
                current = model()->index(rows(current) - 1, current.column(),
                                         rootIndex());
            break;
        default:
            break;
    }

    viewport()->update();
    return current;
}

void ElementViewer::wheelEvent(QWheelEvent *event) {
    if ((event->modifiers() & Qt::ControlModifier) != 0) {
        if (event->delta() > 0) {
            zoomIn();
        } else {
            zoomOut();
        }
    } else
        QAbstractItemView::wheelEvent(event);
}

void ElementViewer::paintEvent(QPaintEvent *event) {

    plask::shared_ptr<ElementWrapper> el = getElementWrapper();
    if (!el) return;

    QPainter painter(viewport());
    //painter.setRenderHint(QPainter::Antialiasing);
    QStyleOptionViewItem option = viewOptions();
    painter.fillRect(event->rect(), option.palette.base());

    painter.save();
    painter.setTransform(getTransformMatrix());
    el->drawReal(painter);

    for (QModelIndex current: selectionModel()->selectedIndexes()) {
       // QItemSelectionModel *selections = ;
       // QModelIndex current = selectionModel()->getcurrentIndex();
        if (current.isValid() && current.parent() == rootIndex()) {
            GeometryTreeItem* i = model()->toItem(current);
            plask::shared_ptr<plask::GeometryElement> e = i->element->wrappedElement;
            if (e->getDimensionsCount() == 2) {
                plask::Box2d b = plask::static_pointer_cast<plask::GeometryElementD<2> >(e)->getBoundingBox();
                QRectF r = toQt(b);
                painter.fillRect(r, QColor(50, 20, 120, 70));
                painter.setPen(QPen(QColor(90, 40, 230, 170), 0.0, Qt::DashLine));
                painter.drawRect(r);
            }
        }
    }

    painter.restore();
}

void ElementViewer::resizeEvent(QResizeEvent * /* event */)
{
    updateGeometries();
}

int ElementViewer::rows(const QModelIndex &index) const
{
    return model()->rowCount(model()->parent(index));
}

void ElementViewer::rowsInserted(const QModelIndex &parent, int start, int end)
{
    /*for (int row = start; row <= end; ++row) {

        QModelIndex index = model()->index(row, 1, rootIndex());
        double value = model()->data(index).toDouble();

        if (value > 0.0) {
            totalValue += value;
            validItems++;
        }
    }*/

    QAbstractItemView::rowsInserted(parent, start, end);
}

void ElementViewer::rowsAboutToBeRemoved(const QModelIndex &parent, int start, int end)
{
    /*for (int row = start; row <= end; ++row) {

        QModelIndex index = model()->index(row, 1, rootIndex());
        double value = model()->data(index).toDouble();
        if (value > 0.0) {
            totalValue -= value;
            validItems--;
        }
    }*/

    QAbstractItemView::rowsAboutToBeRemoved(parent, start, end);
}

void ElementViewer::scrollContentsBy(int dx, int dy)
{
    scrollDirtyRegion(dx, dy);
    viewport()->scroll(dx, dy);
}

void ElementViewer::scrollTo(const QModelIndex &index, ScrollHint)
{
    QRect area = viewport()->rect();
    QRect rect = visualRect(index);

    if (rect.left() < area.left())
        horizontalScrollBar()->setValue(
            horizontalScrollBar()->value() + rect.left() - area.left());
    else if (rect.right() > area.right())
        horizontalScrollBar()->setValue(
            horizontalScrollBar()->value() + qMin(
                rect.right() - area.right(), rect.left() - area.left()));

    if (rect.top() < area.top())
        verticalScrollBar()->setValue(
            verticalScrollBar()->value() + rect.top() - area.top());
    else if (rect.bottom() > area.bottom())
        verticalScrollBar()->setValue(
            verticalScrollBar()->value() + qMin(
                rect.bottom() - area.bottom(), rect.top() - area.top()));

    update();
}

void ElementViewer::setSelection(const QRect &rect, QItemSelectionModel::SelectionFlags command)
{
    plask::Box2d model_sel = fromQt(getTransformMatrix().inverted().mapRect(QRectF(rect)));

    QModelIndexList indexes;

    plask::shared_ptr<plask::GeometryElementD<2> > e = getElement();
    const std::size_t ch_count = e->getRealChildrenCount();
    for (std::size_t i = 0; i < ch_count; ++i) {
        plask::shared_ptr<plask::GeometryElementD<2> > ch = e->getRealChildAt(i)->asD<2>();
        if (ch && ch->getBoundingBox().intersect(model_sel))
            indexes.append(model()->index(i, 0, rootIndex()));
    }

    if (indexes.size() > 0) {
        int firstRow = indexes[0].row();
        int lastRow = indexes[0].row();
        int firstColumn = indexes[0].column();
        int lastColumn = indexes[0].column();

        for (int i = 1; i < indexes.size(); ++i) {
            firstRow = qMin(firstRow, indexes[i].row());
            lastRow = qMax(lastRow, indexes[i].row());
            firstColumn = qMin(firstColumn, indexes[i].column());
            lastColumn = qMax(lastColumn, indexes[i].column());
        }

        QItemSelection selection(
            model()->index(firstRow, firstColumn, rootIndex()),
            model()->index(lastRow, lastColumn, rootIndex()));
        selectionModel()->select(selection, command);
    } else {
        QModelIndex noIndex;
        QItemSelection selection(noIndex, noIndex);
        selectionModel()->select(selection, command);
    }

    update();
}

void ElementViewer::updateGeometries()
{
    plask::Vec<2, double> s = getBoundingBox().size();
    horizontalScrollBar()->setPageStep(viewport()->width());
    horizontalScrollBar()->setRange(0, qMax(0, int( 2* margin + s.c0 * zoom.x() - viewport()->width()) ));
    verticalScrollBar()->setPageStep(viewport()->height());
    verticalScrollBar()->setRange(0, qMax(0, int( 2* margin + s.c1 * zoom.y() - viewport()->height()) ));
}

QTransform ElementViewer::getTransformMatrix() const {
    plask::Box2d bb = getBoundingBox();

    return QTransform(zoom.x(), 0.0,
                      0.0, -zoom.y(),
                      margin - bb.lower.c0 * zoom.x() - horizontalScrollBar()->value(),
                      margin + bb.upper.c1 * zoom.y() - verticalScrollBar()->value());
}


int ElementViewer::verticalOffset() const
{
    return verticalScrollBar()->value();
}

QRect ElementViewer::visualRect(const QModelIndex &index) const
{
    QRectF rect = itemRect(index);
    if (rect.isValid())
        return getTransformMatrix().mapRect(rect).toRect().adjusted(-1, -1, 1, 1);

    return QRect();
}

QRegion ElementViewer::visualRegionForSelection(const QItemSelection &selection) const
{
    int ranges = selection.count();

    if (ranges == 0)
        return QRect();

    QRegion region;
    for (int i = 0; i < ranges; ++i) {
        QItemSelectionRange range = selection.at(i);
        for (int row = range.top(); row <= range.bottom(); ++row) {
            for (int col = range.left(); col <= range.right(); ++col) {
                QModelIndex index = model()->index(row, col, rootIndex());
                region += visualRect(index);
            }
        }
    }
    return region;
}
