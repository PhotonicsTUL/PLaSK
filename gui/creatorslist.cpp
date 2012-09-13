#include "creatorslist.h"

#include "modelext/creator.h"

#include <QStringList>
#include <QMimeData>

int CreatorsListModel::rowCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : getCreators().size();
}

QVariant CreatorsListModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid()) return QVariant();

    switch (role) {
        //case Qt::DecorationRole: return item->icon();
        case Qt::DisplayRole: return QString(getCreators()[index.row()]->getName().c_str());
    }

    return QVariant();

}

QStringList CreatorsListModel::mimeTypes() const {
    return QStringList() << MIME_PTR_TO_CREATOR;
}

QMimeData* CreatorsListModel::mimeData(const QModelIndexList &indexes) const
{
    QMimeData *mimeData = new QMimeData();
    QByteArray encodedData;

    QDataStream stream(&encodedData, QIODevice::WriteOnly);
    foreach (QModelIndex index, indexes) {
        if (index.isValid()) {
            /*QPixmap pixmap = qvariant_cast<QPixmap>(data(index, Qt::UserRole));
            QPoint location = data(index, Qt::UserRole+1).toPoint();
            stream << pixmap << location;*/
            const GeometryObjectCreator* to_write = getCreators()[index.row()];
            stream.writeRawData(reinterpret_cast<const char*>(&to_write), sizeof(to_write));
        }
    }

    mimeData->setData(MIME_PTR_TO_CREATOR, encodedData);
    return mimeData;
}

Qt::ItemFlags CreatorsListModel::flags(const QModelIndex &index) const {
    return QAbstractListModel::flags(index) | Qt::ItemIsDragEnabled;
}
