#include "creatorslist.h"

#include "modelext/creator.h"

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
