#include <math.h>
#include <QtGui>

#include "objectview.h"


#include "../modelext/converter.h"

ObjectViewer::ObjectViewer(QWidget *parent)
    : QAbstractItemView(parent), zoom(10.0, 10.0)
{
    setAcceptDrops(true);

    horizontalScrollBar()->setRange(0, 0);
    verticalScrollBar()->setRange(0, 0);

    margin = 8;
    rubberBand = 0;
}

void ObjectViewer::dragEnterEvent(QDragEnterEvent *event)
{
    if (!rootIndex().isValid()) { event->ignore(); return; }

    if (event->source() != 0 && event->mimeData()->hasFormat(MIME_PTR_TO_CREATOR))
        event->accept();
    else
        event->ignore();
}

void ObjectViewer::dragLeaveEvent(QDragLeaveEvent *event)
{
    QRect updateRect = highlightedRectView();
    highlightedRect = QRectF();
    update(updateRect);
    event->accept();
}

void ObjectViewer::dragMoveEvent(QDragMoveEvent *event)
{
    if (!rootIndex().isValid()) { event->ignore(); return; }

    if (event->source() != 0 && event->mimeData()->hasFormat(MIME_PTR_TO_CREATOR)
        /*&& findPiece(targetSquare(event->pos())) == -1*/) {

        GeometryObjectCreator* creator = GeometryObjectCreator::fromMimeData(event->mimeData());
        highlightedRect = toQt(model()->insertPlace2D(*creator, rootIndex(), fromQt(viewToModel().map(QPointF(event->pos())))));

        event->setDropAction(Qt::MoveAction);
        event->accept();
    } else {
        highlightedRect = QRectF();
        event->ignore();
    }

    viewport()->update();
}

void ObjectViewer::dropEvent(QDropEvent *event)
{
    if (!rootIndex().isValid()) { event->ignore(); return; }

    if (event->source() != 0 &&     //source is local
        event->mimeData()->hasFormat(MIME_PTR_TO_CREATOR)
        /*&& findPiece(targetSquare(event->pos())) == -1*/) {

        GeometryObjectCreator* creator = GeometryObjectCreator::fromMimeData(event->mimeData());

        highlightedRect = QRect();
        viewport()->update();

        event->setDropAction(Qt::CopyAction);
        event->accept();

        int index = model()->insertRow2D(*creator, rootIndex(), fromQt(viewToModel().map(QPointF(event->pos()))));
        if (index != -1)
            selectionModel()->select(model()->index(index, 0, rootIndex()), QItemSelectionModel::ClearAndSelect);

    } else {
        highlightedRect = QRect();
        event->ignore();
    }
}

void ObjectViewer::dataChanged(const QModelIndex &topLeft,
                          const QModelIndex &bottomRight)
{
    QAbstractItemView::dataChanged(topLeft, bottomRight);
    updateGeometries(); //bounding-box could be changed
    viewport()->update();   //redraw
}

bool ObjectViewer::edit(const QModelIndex &index, EditTrigger trigger, QEvent *event)
{
    /*if (index.column() == 0)
        return QAbstractItemView::edit(index, trigger, event);
    else*/
        return false;
}

QModelIndex ObjectViewer::indexAt(const QPoint &point) const
{
    plask::shared_ptr<plask::GeometryObjectD<2> > e = getObject();
    if (!e) return QModelIndex();

    plask::Vec<2, double> model_point = fromQt(viewToModel().map(QPointF(point)));

    const std::size_t ch_count = e->getRealChildrenCount();
    for (std::size_t i = 0; i < ch_count; ++i) {
        plask::shared_ptr<plask::GeometryObjectD<2> > ch = e->getRealChildAt(i)->asD<2>();
        if (ch && ch->getBoundingBox().includes(model_point))
            return model()->index(i, 0, rootIndex());
    }

    return QModelIndex();
}

plask::shared_ptr<ObjectWrapper> ObjectViewer::getObjectWrapper() const
{
    QModelIndex toDrawIndex = rootIndex();
    if (!toDrawIndex.isValid()) return plask::shared_ptr<ObjectWrapper>();    //root or invalidate (deleted) index
    return model()->toItem(toDrawIndex)->getLowerWrappedObject();
}

bool ObjectViewer::isIndexHidden(const QModelIndex & /*index*/) const
{
    return false;
}

QRectF ObjectViewer::itemRect(const QModelIndex &index) const
{
    if (rootIndex().isValid() && index.isValid() && index.parent() == rootIndex()) {
        GeometryTreeItem* i = model()->toItem(index);
        auto e = i->object->wrappedObject;
        if (e->getDimensionsCount() == 2) {
            plask::Box2D b = static_cast< plask::GeometryObjectD<2>* >(e)->getBoundingBox();
            return toQt(b);
        }
    }
    return QRectF();
}

int ObjectViewer::horizontalOffset() const
{
    return horizontalScrollBar()->value();
}

void ObjectViewer::mousePressEvent(QMouseEvent *event)
{
    QAbstractItemView::mousePressEvent(event);
    origin = event->pos();
    if (!rubberBand)
        rubberBand = new QRubberBand(QRubberBand::Rectangle, viewport());
    rubberBand->setGeometry(QRect(origin, QSize()));
    rubberBand->show();
}

void ObjectViewer::mouseMoveEvent(QMouseEvent *event)
{
    if (rubberBand)
        rubberBand->setGeometry(QRect(origin, event->pos()).normalized());
    QAbstractItemView::mouseMoveEvent(event);
}

void ObjectViewer::mouseReleaseEvent(QMouseEvent *event)
{
    QAbstractItemView::mouseReleaseEvent(event);
    if (rubberBand)
        rubberBand->hide();
    viewport()->update();
}

QModelIndex ObjectViewer::moveCursor(QAbstractItemView::CursorAction cursorAction,
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

void ObjectViewer::wheelEvent(QWheelEvent *event) {
    if ((event->modifiers() & Qt::ControlModifier) != 0) {
        if (event->delta() > 0) {
            zoomIn();
        } else {
            zoomOut();
        }
    } else
        QAbstractItemView::wheelEvent(event);
}

void ObjectViewer::paintEvent(QPaintEvent *event) {

    plask::shared_ptr<ObjectWrapper> el = getObjectWrapper();
    if (!el) return;

    QPainter painter(viewport());
    //painter.setRenderHint(QPainter::Antialiasing);
    QStyleOptionViewItem option = viewOptions();
    painter.fillRect(event->rect(), option.palette.base());

    painter.save();
    painter.setTransform(modelToView());
    el->drawReal(painter);

    for (QModelIndex current: selectionModel()->selectedIndexes()) {
       // QItemSelectionModel *selections = ;
       // QModelIndex current = selectionModel()->getcurrentIndex();
        if (current.isValid() && current.parent() == rootIndex()) {
            GeometryTreeItem* i = model()->toItem(current);
            plask::GeometryObject* e = i->object->wrappedObject;
            if (e->getDimensionsCount() == 2) {
                plask::Box2D b = static_cast<plask::GeometryObjectD<2>*>(e)->getBoundingBox();
                QRectF r = toQt(b);
                painter.fillRect(r, QColor(50, 20, 120, 70));
                painter.setPen(QPen(QColor(90, 40, 230, 170), 0.0, Qt::DashLine));
                painter.drawRect(r);
            }
        }
    }

    if (highlightedRect.isValid()) {
        painter.fillRect(highlightedRect, QColor(120, 20, 50, 70));
        painter.setPen(QPen(QColor(230, 40, 90, 170), 0.3, Qt::SolidLine));
        painter.drawRect(highlightedRect);
    }

    painter.restore();
}

void ObjectViewer::resizeEvent(QResizeEvent * /* event */)
{
    updateGeometries();
}

int ObjectViewer::rows(const QModelIndex &index) const
{
    return model()->rowCount(model()->parent(index));
}

void ObjectViewer::rowsInserted(const QModelIndex &parent, int start, int end)
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

void ObjectViewer::rowsAboutToBeRemoved(const QModelIndex &parent, int start, int end)
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

void ObjectViewer::scrollContentsBy(int dx, int dy)
{
    scrollDirtyRegion(dx, dy);
    viewport()->scroll(dx, dy);
}

void ObjectViewer::scrollTo(const QModelIndex &index, ScrollHint)
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

void ObjectViewer::setSelection(const QRect &rect, QItemSelectionModel::SelectionFlags command)
{
    plask::Box2D model_sel = fromQt(viewToModel().mapRect(QRectF(rect)));

    QModelIndexList indexes;

    plask::shared_ptr<plask::GeometryObjectD<2> > e = getObject();

    if (e) {
        const std::size_t ch_count = e->getRealChildrenCount();
        for (std::size_t i = 0; i < ch_count; ++i) {
            plask::shared_ptr<plask::GeometryObjectD<2> > ch = e->getRealChildAt(i)->asD<2>();
            if (ch && ch->getBoundingBox().intersects(model_sel))
                indexes.append(model()->index(i, 0, rootIndex()));
        }
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

void ObjectViewer::updateGeometries()
{
    plask::Vec<2, double> s = getBoundingBox().size();
    horizontalScrollBar()->setPageStep(viewport()->width());
    horizontalScrollBar()->setRange(0, qMax(0, int( 2* margin + s.c0 * zoom.x() - viewport()->width()) ));
    verticalScrollBar()->setPageStep(viewport()->height());
    verticalScrollBar()->setRange(0, qMax(0, int( 2* margin + s.c1 * zoom.y() - viewport()->height()) ));
}

QTransform ObjectViewer::modelToView() const {
    plask::Box2D bb = getBoundingBox();

    return QTransform(zoom.x(), 0.0,
                      0.0, -zoom.y(),
                      margin - bb.lower.c0 * zoom.x() - horizontalScrollBar()->value(),
                      margin + bb.upper.c1 * zoom.y() - verticalScrollBar()->value());
}


int ObjectViewer::verticalOffset() const
{
    return verticalScrollBar()->value();
}

QRect ObjectViewer::visualRect(const QModelIndex &index) const
{
    QRectF rect = itemRect(index);
    if (rect.isValid())
        return modelToView().mapRect(rect).toRect().adjusted(-1, -1, 1, 1);

    return QRect();
}

QRegion ObjectViewer::visualRegionForSelection(const QItemSelection &selection) const
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

