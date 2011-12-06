
#include "../vector/2d.h"
#include "../vector/3d.h"

namespace plask {

struct Rect2d {
    
        Vec2<double> lower;
        
        Vec2<double> upper;
        
        Vec2<double> size() const { return upper - lower; }
        
        bool inside(const Vec2<double>& p) {
        }
        
        bool overlap(const Rect2d& other) const {
        };
    
};

}       // namespace plask
