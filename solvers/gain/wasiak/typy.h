#ifndef _TYPY_H
#define _TYPY_H

enum rodzaj {ciezka, lekka};

template <class typ> typ mniej(typ a,typ b)
{
 return (a<b)?a:b;
}
#endif
