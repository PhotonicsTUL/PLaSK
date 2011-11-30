#ifndef PLASK__INTERPOLATION_H
#define PLASK__INTERPOLATION_H

#include <typeinfo>  // for 'typeid'

#include "mesh.h"
#include "plask/exceptions.h"

namespace plask {

/**
Supported interpolation methods.
@see @ref meshes_interpolation
*/
enum InterpolationMethod {
    DEFAULT = 0,        ///< default interpolation (depends on source mesh)
    LINEAR = 1,         ///< linear interpolation
    SPLINE = 2,         ///< spline interpolation
    //...add new interpolation algoritms here...
    __ILLEGAL_INTERPOLATION_METHOD__  // necessary for metaprogram loop
};

static const char* InterpolationMethodNames[] = { "DEFAULT", "LINEAR", "SPLINE" /*attach new interpolation algoritm names here*/};

/**
Specialization of this class are used for interpolation and can depend on source mesh type,
data type and the interpolation method.
@see @ref interpolation_write
*/
template <typename SrcMeshT, typename DataT, InterpolationMethod method>
struct InterpolationAlgorithm
{
    static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec, const Mesh& dst_mesh, std::vector<DataT>& dst_vec) {
        std::string msg = "interpolate (source mesh type: ";
        msg += typeid(src_mesh).name();
        msg += ", interpolation method: ";
        msg += InterpolationMethodNames[method];
        msg += ")";
        throw NotImplemented(msg);
        //TODO iterate over dst_mesh and call InterpolationAlgorithmForPoint
    }
};


// The following structures are solely used for metaprogramming
template <typename SrcMeshT, typename DataT, int iter>
struct __InterpolateMeta__
{
    inline static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec,
                Mesh& dst_mesh, std::vector<DataT>& dst_vec, InterpolationMethod method) {
        if (int(method) == iter)
            InterpolationAlgorithm<SrcMeshT, DataT, (InterpolationMethod)iter>::interpolate(src_mesh, src_vec, dst_mesh, dst_vec);
        else
            __InterpolateMeta__<SrcMeshT, DataT, iter+1>::interpolate(src_mesh, src_vec, dst_mesh, dst_vec, method);
    }
};
template <typename SrcMeshT, typename DataT>
struct __InterpolateMeta__<SrcMeshT, DataT, __ILLEGAL_INTERPOLATION_METHOD__>
{
    inline static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec,
                Mesh& dst_mesh, std::vector<DataT>& dst_vec, InterpolationMethod method) {
        throw CriticalException("No such interpolation method.");
    }
};


/**
Calculate (interpolate when needed) a field of some physical properties in requested points of (@a dst_mesh)
if values of this field in points of (@a src_mesh) are known.
@param src_mesh set of points in which fields values are known
@param src_vec vector of known field values in points described by @a sec_mesh
@param dst_mesh requested set of points, in which the field values should be calculated (interpolated)
@param method interpolation method to use
@return vector of the field values in points described by @a dst_mesh, can be equal to @a src_vec
        if @a src_mesh and @a dst_mesh are the same mesh
@throw NotImplemented if given interpolation method is not implemented for used source mesh type
@throw CriticalException if given interpolation method is not valid
@see @ref meshes_interpolation
*/
template <typename SrcMeshT, typename DataT>
inline std::shared_ptr<const std::vector<DataT>>
translateField(SrcMeshT& src_mesh, std::shared_ptr<const std::vector<DataT>>& src_vec, Mesh& dst_mesh,
               InterpolationMethod method = DEFAULT) {

    if (&src_mesh == &dst_mesh) return src_vec; // meshes are identical, so just return src_vec

    if (dynamic_cast<MeshOver<typename SrcMeshT::Space>*>(&dst_mesh)) {
        // Both meshes are over the same space
        //interpolate without translating to global coordinates
    } else {
        //interpolate with translating to global coordinates
    }

    std::shared_ptr<std::vector<DataT>> result(new std::vector<DataT>);
    __InterpolateMeta__<SrcMeshT, DataT, 0>::interpolate(src_mesh, *src_vec, dst_mesh, *result, method);
    return result;
}

/*
    Ponieważ wszystkie prawdziwe mesze muszą dziedziczyć po MeshOver, można używając bardziej rozbudowany
    dynamic_cast wyciągnąć typ przestrzeni na jakiej jest rozpięta siatka docelowa. Idąc dalej można na przykład
    wymusić by funkcje interpolujące uwzględniały typ przestrzeni (jako dodatkowy parametr szablonu).

    Inną możliwością jest dodanie innej grupy funkcji, które przepisują punkty z jednej przestrzeni do drugiej,
    a interpolatory działają zawsze w jednej przestrzeni (czyli dst_mesh jest typu MeshOver<SrcMeshT::Space>).
    Druga grupa funkcji tłumaczy przestrzenie (może wystarczą szablony wykorzystujące metody klas Space###).
    Do tego można się zastanowić, czy do konkretyzacji klas MeshOver<> nie dodać jakichś mechanizmów uśredniania
    jeżeli wykonywana jest translacja z przestrzeni 3D do 2D (oczywiście w siatce powinno być jakoś zapisane po
    ilu i których punktach uśredniamy). Dobrze by było też zapewnić by próba translacji pomiędzy SpaceXY i SpaceRZ
    wywaliła wyjątek (one naprawdę nie mają sensu jednocześnie).

    Zupełnie nieporuszonym tematem są siatki będące przekrojami (na przykład siatka 1D mająca znaczenie tylko
    w obszarze czynnym). Logiczne jest jednak by przekrój przynależał do przestrzeni całości (czyli siatka 1D
    będąca przekrojem obszaru czynnego, tak naprawdę przynależy do przestrzeni SpaceXY lub SpaceRZ — i tak powinny
    być dwie niezależne, bo równania będą wyglądać inaczej).

    Kolejnym tematem jest translacja wartości wektorowych pomiędzy przestrzeniami. Może to powodować, że src_vec
    i dst_vec będą różnych typów. Pytanie kto i w którym momencie powinno robić to przeliczenie. Provider? Rzecz
    w tym, że to czy to przeliczenie jest potrzebnie można wiedzieć porównując przestrzenie (jak są różne to jest
    potrzebne), a nie możemy przewidzieć jakie moduły zostaną połączone w parę Provider-Receiver. Na pewno nie jest
    opcją robienie przeliczenia zawsze, bo to jest za duże marnotrawstwo czasu i pamięci jeżeli przestrzenie są
    takie same. PROPOZYCJA_1: Napisać oddzielną funkcję traslateVectorField, analogiczną do TranslateField, która 
    robi to samo co translateField, a do tego przekształca odpowiednio składowe wektorów. Oczywiście w niej wiemy
    i wymuszamy, by DataT było zawsze Vec3<T> lub Vec2<T>. Zwracane wektory byłyby zawsze Vec3 w przestrzeni ABC.
    Moduł receivera sam by sobie je przeliczał do swojej przestrzeni. Tylko pewnie wtedy moduły dostarczające
    wartości wektorowe musiały by mieć dla nich dwa różne providery i to użytkownik programu musiałbym podłączyć
    ten właściwy (inaczej otrzyma bzdurne wyniki). PROPOZYCJA_2: robimy klasy dziedziczące po Vec2 i Vec3, które już
    jasno określają swoją przestrzeń i piszemy odpowiednie operatory castowania (opierając się na metodach w Space###).
    Tych przestrzeni jest niewiele, więc może nie będzie to taka głupia metoda (na chwilę obecną wydaje mi się
    najbardziej sensowna). Oczywiście twórca modułu będzie musiał zadbać by typ danych wektorowych pasował do
    przestrzeni, na której pracuje (można mu ułatwić życie definując odpowiednie typy w MeshOver<###>).
*/

} // namespace plask

#endif  //PLASK__INTERPOLATION_H
