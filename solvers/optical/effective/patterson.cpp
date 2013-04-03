#include "patterson.h"

namespace plask { namespace solvers { namespace effective {

dcomplex patterson(const std::function<dcomplex(dcomplex)>& fun, dcomplex a, dcomplex b, double eps)
{
    const static double points[] = {
        0.000000000000000,
        0.007047138459337,
        0.014093886410782,
        0.021139853378331,
        0.028184648949746,
        0.035227882808441,
        0.042269164765364,
        0.049308104790869,
        0.056344313046593,
        0.063377399917322,
        0.070406976042855,
        0.077432652349857,
        0.084454040083711,
        0.091470750840355,
        0.098482396598119,
        0.105488589749542,
        0.112488943133187,
        0.119483070065440,
        0.126470584372302,
        0.133451100421162,
        0.140424233152560,
        0.147389598111940,
        0.154346811481378,
        0.161295490111305,
        0.168235251552207,
        0.175165714086311,
        0.182086496759252,
        0.188997219411722,
        0.195897502711100,
        0.202786968183065,
        0.209665238243181,
        0.216531936228473,
        0.223386686428967,
        0.230229114119222,
        0.237058845589830,
        0.243875508178893,
        0.250678730303483,
        0.257468141491070,
        0.264243372410927,
        0.271004054905513,
        0.277749822021824,
        0.284480308042726,
        0.291195148518247,
        0.297893980296858,
        0.304576441556714,
        0.311242171836872,
        0.317890812068477,
        0.324522004605922,
        0.331135393257977,
        0.337730623318886,
        0.344307341599438,
        0.350865196458001,
        0.357403837831532,
        0.363922917266550,
        0.370422087950078,
        0.376901004740559,
        0.383359324198730,
        0.389796704618471,
        0.396212806057616,
        0.402607290368737,
        0.408979821229889,
        0.415330064175322,
        0.421657686626163,
        0.427962357921063,
        0.434243749346803,
        0.440501534168876,
        0.446735387662028,
        0.452944987140767,
        0.459130011989832,
        0.465290143694635,
        0.471425065871659,
        0.477534464298829,
        0.483618026945841,
        0.489675444004456,
        0.495706407918761,
        0.501710613415392,
        0.507687757533717,
        0.513637539655989,
        0.519559661537457,
        0.525453827336443,
        0.531319743644376,
        0.537157119515795,
        0.542965666498311,
        0.548745098662529,
        0.554495132631933,
        0.560215487612728,
        0.565905885423654,
        0.571566050525743,
        0.577195710052046,
        0.582794593837319,
        0.588362434447663,
        0.593898967210122,
        0.599403930242243,
        0.604877064481584,
        0.610318113715186,
        0.615726824608993,
        0.621102946737226,
        0.626446232611720,
        0.631756437711194,
        0.637033320510492,
        0.642276642509759,
        0.647486168263572,
        0.652661665410017,
        0.657802904699714,
        0.662909660024781,
        0.667981708447750,
        0.673018830230419,
        0.678020808862644,
        0.682987431091079,
        0.687918486947839,
        0.692813769779115,
        0.697673076273711,
        0.702496206491527,
        0.707282963891961,
        0.712033155362252,
        0.716746591245747,
        0.721423085370099,
        0.726062455075390,
        0.730664521242181,
        0.735229108319492,
        0.739756044352695,
        0.744245161011347,
        0.748696293616937,
        0.753109281170558,
        0.757483966380514,
        0.761820195689839,
        0.766117819303760,
        0.770376691217077,
        0.774596669241483,
        0.778777615032823,
        0.782919394118283,
        0.787021875923539,
        0.791084933799848,
        0.795108445051101,
        0.799092290960841,
        0.803036356819269,
        0.806940531950218,
        0.810804709738147,
        0.814628787655137,
        0.818412667287926,
        0.822156254364980,
        0.825859458783650,
        0.829522194637401,
        0.833144380243173,
        0.836725938168869,
        0.840266795261030,
        0.843766882672709,
        0.847226135891581,
        0.850644494768350,
        0.854021903545469,
        0.857358310886232,
        0.860653669904300,
        0.863907938193690,
        0.867121077859315,
        0.870293055548114,
        0.873423842480859,
        0.876513414484705,
        0.879561752026556,
        0.882568840247342,
        0.885534668997285,
        0.888459232872257,
        0.891342531251320,
        0.894184568335559,
        0.896985353188317,
        0.899744899776940,
        0.902463227016166,
        0.905140358813262,
        0.907776324115059,
        0.910371156957004,
        0.912924896514371,
        0.915437587155765,
        0.917909278499078,
        0.920340025470012,
        0.922729888363349,
        0.925078932907076,
        0.927387230329537,
        0.929654857429740,
        0.931881896650954,
        0.934068436157726,
        0.936214569916451,
        0.938320397779593,
        0.940386025573670,
        0.942411565191083,
        0.944397134685867,
        0.946342858373403,
        0.948248866934137,
        0.950115297521295,
        0.951942293872574,
        0.953730006425761,
        0.955478592438184,
        0.957188216109861,
        0.958859048710200,
        0.960491268708020,
        0.962085061904651,
        0.963640621569812,
        0.965158148579916,
        0.966637851558417,
        0.968079947017760,
        0.969484659502459,
        0.970852221732792,
        0.972182874748582,
        0.973476868052507,
        0.974734459752403,
        0.975955916702012,
        0.977141514639706,
        0.978291538324759,
        0.979406281670863,
        0.980486047876721,
        0.981531149553740,
        0.982541908851081,
        0.983518657578633,
        0.984461737328815,
        0.985371499598520,
        0.986248305913008,
        0.987092527954034,
        0.987904547695124,
        0.988684757547429,
        0.989433560520241,
        0.990151370400770,
        0.990838611958294,
        0.991495721178106,
        0.992123145530863,
        0.992721344282789,
        0.993290788851685,
        0.993831963212755,
        0.994345364356723,
        0.994831502800621,
        0.995290903148810,
        0.995724104698407,
        0.996131662079315,
        0.996514145914890,
        0.996872143485260,
        0.997206259372222,
        0.997517116063472,
        0.997805354495957,
        0.998071634524930,
        0.998316635318407,
        0.998541055697168,
        0.998745614468095,
        0.998931050830811,
        0.999098124967668,
        0.999247618943342,
        0.999380338025024,
        0.999497112467187,
        0.999598799671911,
        0.999686286448318,
        0.999760490924432,
        0.999822363679788,
        0.999872888120358,
        0.999913081144678,
        0.999943996207054,
        0.999966730098486,
        0.999982430354892,
        0.999992298136258,
        0.999997596379748,
        0.999999672956734
    };
    const static double weights[][256] = {
        { // n = 1
            2.
        },
        { // n = 3
            0.888888888888889,
            0.555555555555556
        },
        { // n = 7
            0.450916538658474,
            0.401397414775962,
            0.268488089868333,
            0.104656226026467
        },
        { // n = 15
            0.225510499798207,
            0.219156858401587,
            0.200628529376989,
            0.171511909136391,
            0.134415255243784,
            0.0929271953151245,
            0.0516032829970797,
            0.0170017196299403
        },
        { // n = 31
            0.112755256720769,
            0.111956873020953,
            0.109578421055925,
            0.105669893580235,
            0.100314278611796,
            0.0936271099812645,
            0.0857559200499903,
            0.0768796204990035,
            0.0672077542959907,
            0.0569795094941234,
            0.046462893261758,
            0.0359571033071293,
            0.0258075980961767,
            0.0164460498543878,
            0.00843456573932111,
            0.00254478079156187
        },
        { // n = 63
            0.0563776283603847,
            0.0562776998312543,
            0.0559784365104763,
            0.0554814043565594,
            0.0547892105279629,
            0.0539054993352661,
            0.0528349467901165,
            0.0515832539520485,
            0.0501571393058995,
            0.0485643304066732,
            0.046813554990628,
            0.0449145316536322,
            0.0428779600250077,
            0.0407155101169443,
            0.0384398102494555,
            0.0360644327807826,
            0.0336038771482077,
            0.031073551111688,
            0.0284897547458336,
            0.0258696793272147,
            0.0232314466399103,
            0.0205942339159127,
            0.0179785515681283,
            0.0154067504665595,
            0.0129038001003513,
            0.0104982469096213,
            0.00822300795723593,
            0.00611550682211725,
            0.00421763044155885,
            0.00257904979468569,
            0.00126515655623007,
            0.000363221481845531
        },
        { // n = 127
            0.0281888141801924,
            0.0281763190330166,
            0.0281388499156272,
            0.0280764557938172,
            0.0279892182552382,
            0.0278772514766137,
            0.0277407021782797,
            0.0275797495664819,
            0.0273946052639814,
            0.0271855132296248,
            0.026952749667633,
            0.0266966229274504,
            0.0264174733950583,
            0.0261156733767061,
            0.0257916269760242,
            0.0254457699654648,
            0.0250785696529498,
            0.0246905247444877,
            0.0242821652033366,
            0.0238540521060385,
            0.023406777495314,
            0.0229409642293877,
            0.0224572658268161,
            0.0219563663053178,
            0.0214389800125039,
            0.020905851445812,
            0.0203577550584722,
            0.0197954950480975,
            0.0192199051247278,
            0.0186318482561388,
            0.0180322163903913,
            0.0174219301594642,
            0.0168019385741039,
            0.0161732187295777,
            0.015536775555844,
            0.0148936416648152,
            0.0142448773729168,
            0.0135915710097655,
            0.0129348396636074,
            0.0122758305600828,
            0.0116157233199551,
            0.0109557333878379,
            0.0102971169579564,
            0.00964117772970254,
            0.00898927578406414,
            0.00834283875396816,
            0.00770337523327974,
            0.00707248999543356,
            0.00645190005017574,
            0.00584344987583564,
            0.00524912345480886,
            0.00467105037211432,
            0.00411150397865469,
            0.0035728927835173,
            0.00305775341017553,
            0.00256876494379402,
            0.00210881524572663,
            0.00168114286542147,
            0.00128952408261042,
            0.000938369848542382,
            0.000632607319362634,
            0.000377746646326985,
            0.000180739564445388,
            5.05360952078625e-05
        },
        { // n = 255
            0.0140944070900962,
            0.0140928450691604,
            0.0140881595165083,
            0.0140803519625537,
            0.0140694249578136,
            0.01405538207265,
            0.0140382278969086,
            0.0140179680394566,
            0.0139946091276191,
            0.0139681588065169,
            0.0139386257383069,
            0.0139060196013255,
            0.0138703510891398,
            0.0138316319095064,
            0.0137898747832409,
            0.0137450934430019,
            0.0136973026319907,
            0.0136465181025713,
            0.0135927566148124,
            0.0135360359349562,
            0.0134763748338165,
            0.0134137930851101,
            0.0133483114637252,
            0.0132799517439305,
            0.0132087366975291,
            0.0131346900919602,
            0.013057836688353,
            0.0129782022395374,
            0.0128958134880121,
            0.0128106981638774,
            0.0127228849827324,
            0.0126324036435421,
            0.0125392848264749,
            0.012443560190714,
            0.0123452623722438,
            0.012244424981612,
            0.0121410826016683,
            0.0120352707852796,
            0.0119270260530193,
            0.0118163858908302,
            0.011703388747657,
            0.011588074033044,
            0.0114704821146939,
            0.0113506543159806,
            0.011228632913408,
            0.0111044611340069,
            0.0109781831526589,
            0.0108498440893373,
            0.0107194900062519,
            0.0105871679048852,
            0.010452925722906,
            0.0103168123309476,
            0.0101788775292361,
            0.0100391720440568,
            0.00989774752404875,
            0.00975465653631741,
            0.00960995256236388,
            0.00946368999383007,
            0.0093159241280694,
            0.00916671116356079,
            0.00901610819519564,
            0.0088641732094825,
            0.00871096507973209,
            0.00855654356130769,
            0.00840096928705193,
            0.00824430376303287,
            0.00808660936478886,
            0.00792794933429485,
            0.00776838777792199,
            0.00760798966571906,
            0.00744682083240759,
            0.00728494798055381,
            0.00712243868645839,
            0.00695936140939042,
            0.00679578550488277,
            0.00663178124290189,
            0.00646741983180369,
            0.00630277344908576,
            0.00613791528004138,
            0.00597291956550817,
            0.00580786165997757,
            0.00564281810138444,
            0.00547786669391895,
            0.00531308660518706,
            0.00514855847897818,
            0.00498436456476554,
            0.00482058886485127,
            0.00465731729975685,
            0.00449463789203207,
            0.00433264096809298,
            0.00417141937698408,
            0.00401106872407502,
            0.00385168761663987,
            0.00369337791702565,
            0.00353624499771678,
            0.00338039799108692,
            0.00322595002508787,
            0.00307301843470258,
            0.00292172493791782,
            0.00277219576459345,
            0.00262456172740443,
            0.00247895822665757,
            0.00233552518605716,
            0.00219440692536384,
            0.00205575198932735,
            0.00191971297101387,
            0.00178644639175865,
            0.00165611272815445,
            0.00152887670508777,
            0.00140490799565514,
            0.00128438247189701,
            0.00116748411742996,
            0.00105440762286332,
            0.000945361516858525,
            0.000840571432710722,
            0.000740282804244503,
            0.000644762041305725,
            0.000554295314930375,
            0.00046918492424785,
            0.000389745284473282,
            0.000316303660822264,
            0.000249212400482997,
            0.000188873264506505,
            0.000135754910949229,
            9.03727346587511e-05,
            5.32752936697806e-05,
            2.51578703842807e-05,
            6.93793643241083e-06
        },
        { // n = 511
            0.00704720354504809,
            0.00704700828844548,
            0.0070464225345802,
            0.00704544633127951,
            0.00704407975825415,
            0.00704232292709631,
            0.00704017598127683,
            0.00703763909614153,
            0.00703471247890679,
            0.00703139636865429,
            0.00702769103632498,
            0.00702359678471226,
            0.00701911394845431,
            0.00701424289402573,
            0.0070089840197283,
            0.00700333775568107,
            0.00699730456380954,
            0.00699088493783425,
            0.00698407940325847,
            0.0069768885173552,
            0.00696931286915343,
            0.00696135307942367,
            0.00695300980066273,
            0.00694428371707783,
            0.00693517554456992,
            0.00692568603071643,
            0.00691581595475321,
            0.00690556612755588,
            0.00689493739162047,
            0.00688393062104341,
            0.00687254672150095,
            0.00686078663022781,
            0.00684865131599536,
            0.00683614177908911,
            0.00682325905128565,
            0.00681000419582895,
            0.0067963783074062,
            0.00678238251212301,
            0.00676801796747811,
            0.00675328586233753,
            0.00673818741690826,
            0.00672272388271144,
            0.00670689654255505,
            0.00669070671050613,
            0.00667415573186259,
            0.00665724498312455,
            0.00663997587196526,
            0.00662234983720169,
            0.00660436834876457,
            0.00658603290766825,
            0.00656734504598008,
            0.00654830632678944,
            0.00652891834417652,
            0.00650918272318071,
            0.0064891011197687,
            0.00646867522080231,
            0.00644790674400606,
            0.00642679743793437,
            0.00640534908193868,
            0.00638356348613414,
            0.00636144249136619,
            0.0063389879691769,
            0.00631620182177104,
            0.00629308598198199,
            0.00626964241323744,
            0.00624587310952491,
            0.00622178009535702,
            0.00619736542573666,
            0.00617263118612192,
            0.00614757949239084,
            0.00612221249080599,
            0.00609653235797889,
            0.00607054130083415,
            0.00604424155657355,
            0.00601763539263978,
            0.0059907251066801,
            0.00596351302650963,
            0.0059360015100746,
            0.00590819294541512,
            0.00588008975062789,
            0.0058516943738285,
            0.00582300929311348,
            0.00579403701652198,
            0.00576478008199711,
            0.00573524105734694,
            0.00570542254020497,
            0.0056753271579903,
            0.00564495756786715,
            0.00561431645670402,
            0.00558340654103216,
            0.00555223056700346,
            0.00552079131034779,
            0.00548909157632946,
            0.0054571341997031,
            0.00542492204466866,
            0.00539245800482556,
            0.00535974500312597,
            0.00532678599182712,
            0.0052935839524426,
            0.00526014189569259,
            0.00522646286145301,
            0.00519254991870342,
            0.00515840616547381,
            0.00512403472879005,
            0.00508943876461804,
            0.0050546214578065,
            0.00501958602202842,
            0.00498433569972103,
            0.00494887376202437,
            0.00491320350871842,
            0.00487732826815871,
            0.00484125139721057,
            0.00480497628118194,
            0.00476850633375475,
            0.00473184499691503,
            0.00469499574088179,
            0.0046579620640347,
            0.00462074749284081,
            0.00458335558178039,
            0.00454578991327213,
            0.00450805409759782,
            0.00447015177282693,
            0.00443208660474125,
            0.00439386228676004,
            0.00435548253986604,
            0.0043169511125328,
            0.00427827178065384,
            0.00423944834747438,
            0.00420048464352597,
            0.0041613845265651,
            0.00412215188151643,
            0.00408279062042158,
            0.00404330468239443,
            0.00400369803358422,
            0.00396397466714742,
            0.00392413860322996,
            0.003884193888961,
            0.00384414459846013,
            0.00380399483285953,
            0.00376374872034296,
            0.0037234104162038,
            0.00368298410292404,
            0.0036424739902769,
            0.00360188431545532,
            0.00356121934322919,
            0.00352048336613418,
            0.00347968070469521,
            0.00343881570768791,
            0.00339789275244139,
            0.00335691624518617,
            0.00331589062145094,
            0.00327482034651234,
            0.00323370991590184,
            0.00319256385597435,
            0.00315138672454288,
            0.00311018311158428,
            0.00306895764002069,
            0.00302771496658199,
            0.00298645978275408,
            0.00294519681581858,
            0.00290393082998878,
            0.00286266662764758,
            0.00282140905069222,
            0.00278016298199139,
            0.00273893334695948,
            0.00269772511525295,
            0.00265654330259353,
            0.00261539297272236,
            0.00257427923948909,
            0.00253320726907925,
            0.00249218228238277,
            0.00245120955750556,
            0.00241029443242563,
            0.00236944230779381,
            0.00232865864987843,
            0.00228794899365196,
            0.00224731894601603,
            0.00220677418916003,
            0.00216632048404649,
            0.00212596367401473,
            0.00208570968849204,
            0.00204556454679958,
            0.00200553436203751,
            0.00196562534503151,
            0.00192584380831994,
            0.00188619617015808,
            0.00184668895851283,
            0.00180732881501809,
            0.00176812249885839,
            0.00172907689054462,
            0.00169019899554346,
            0.00165149594771915,
            0.00161297501254393,
            0.00157464359003212,
            0.00153650921735129,
            0.00149857957106457,
            0.00146086246895891,
            0.00142336587141721,
            0.00138609788229673,
            0.00134906674928353,
            0.00131228086370221,
            0.00127574875977347,
            0.00123947911332878,
            0.00120348074001266,
            0.00116776259302858,
            0.00113233376051598,
            0.00109720346268192,
            0.0010623810488534,
            0.00102787599466367,
            0.000993697899638761,
            0.000959856485506936,
            0.000926361595613111,
            0.000893223195879325,
            0.000860451377808528,
            0.000828056364077226,
            0.000796048517297551,
            0.000764438352543883,
            0.000733236554224768,
            0.000702453997827572,
            0.000672101776960108,
            0.000642191235948505,
            0.000612734008012225,
            0.00058374205871498,
            0.000555227733977308,
            0.000527203811431658,
            0.000499683553312801,
            0.000472680758429263,
            0.000446209810101403,
            0.000420285716355361,
            0.000394924138246874,
            0.000370141402122252,
            0.000345954492129904,
            0.000322381020652862,
            0.000299439176850912,
            0.000277147657465187,
            0.000255525589595237,
            0.000234592462123925,
            0.000214368090034217,
            0.000194872642236641,
            0.000176126765545083,
            0.000158151830411132,
            0.000140970302204105,
            0.000124606200241498,
            0.000109085545645742,
            9.44366322532705e-05,
            8.06899228014035e-05,
            6.78774554733972e-05,
            5.60319507856164e-05,
            4.51863674126296e-05,
            3.5375137205519e-05,
            2.66376412339001e-05,
            1.90213681905876e-05,
            1.25792781889593e-05,
            7.36624069102322e-06,
            3.45456507169149e-06,
            9.45715933950007e-07
        }
    };

    double err;

    dcomplex result, result2;
    dcomplex D = (b - a) / 2., Z = (a + b) / 2.;

    dcomplex values[511]; std::fill_n(values, 511, 0.);
    values[0] = fun(D);

    for (unsigned n = 0; n < 2 || (err > eps && n < 9); ++n) {
        unsigned N = (2 << n) - 1; // number of point in current iteration
        result2 = result;
        result = 0.;
        for (unsigned i = 0; i < N; ++i) {
//             double x = points[];
//             dcomplex z1 = Z + D * x;
//             dcomplex z2 = Z - D * x;
        }
    }


}

}}} // namespace plask::solvers::effective
