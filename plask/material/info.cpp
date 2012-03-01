#include "info.h"

namespace plask {

MaterialInfoDB& MaterialInfoDB::getDefault() {
    static MaterialInfoDB defaultInfoDB;
    return defaultInfoDB;
}

}
