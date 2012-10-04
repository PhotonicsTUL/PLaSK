#ifndef PLASK__FUNCTIONS_H
#define PLASK__FUNCTIONS_H

namespace plask
{
    namespace cFunc // functions
	{
		const double Varshni(double iEg0K, double iAlpha, double iBeta, double iT)
		{
			return (iEg0K - iAlpha * iT * iT / (iT + iBeta)); // [K]
		}
	}
} // namespace

#endif
