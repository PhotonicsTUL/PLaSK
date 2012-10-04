#include "functions.h"

namespace plask
{
    namespace cFunc // functions
    {
        double Varshni(double iEg0K, double iAlpha, double iBeta, double iT)
		{
			return (iEg0K - iAlpha * iT * iT / (iT + iBeta)); // [K]
		}
    }
} // namespace

