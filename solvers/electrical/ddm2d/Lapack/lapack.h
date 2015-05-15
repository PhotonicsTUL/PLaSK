/*
    To jest plik nagłówkowy do funkcji LAPACKa rozwiązującej układ równań
    z macierzą liczb rzeczywistych (double) symetryczna i pasmową.

    N to wymiar macierzy, kd ilość pasm pod przekątną.

    Macierz A musi byś symetryczna, pasmowa i dodatnio określona.

    Macierz A powinna być przechowywana w tablicy A[N*(kd+1)]

    Odwzorowanie mapowania A[i,j] -> A[i+kd*j]
    o ile j <= i (jak nie to zamienić je) oraz indeksowanie jest od zera

    dla N = 5, k = 2:
    współrzędne:                 indeksy:
    | A00 a10 a20         |      | 00 __ __       |
    | A10 A11 a21 a31     |      | 01 03 __ __    |
    | A20 A21 A22 a32 a42 |      | 02 04 06 __ __ |
    |     A31 A32 A33 a43 |      |    05 07 09 __ |
    |         A42 A43 A44 |      |       08 10 12 |
                   *   *                    11 13
                       *                       14

    Macierz B to zwykła macierz zawierająca wektory wyrazów wolnych.
    Zapisane sa one kolumnowo, tj. B[i,j] -> B[i+N*j]

    nb to ilość kolumn macierzy B (1 jeżeli tylko jedno równanie)

    Rozwiązujemy: A * X = B

    Rozwiązanie zapisywane jest do B.
*/

#ifndef MD_LAPACK_H
#define MD_LAPACK_H

// To jest funkcja z LAPACKa
extern "C" {
    void dpbsv_(const char& uplo, const int& N, const int& kd, const int& nrhs, double* A, const int& ldab, double* B, const int& ldb, int& info);
};


// To jest wrapper do łatwiejszego wykorzystania
inline int SolveEquation(double* A, double* B, int N, int kd, int nb=1) {
    int info;
    dpbsv_('L', N, kd, nb, A, kd+1, B, N, info);
    return info; // jeżeli info = 0 to jest ok
}

#endif