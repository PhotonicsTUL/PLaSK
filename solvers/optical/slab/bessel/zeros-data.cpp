/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "zeros-data.hpp"

namespace plask { namespace optical { namespace slab {

const double bessel_zeros[][100] = {
    {   // m = 0
        2.404825557695773,
        5.520078110286311,
        8.653727912911011,
        11.79153443901428,
        14.93091770848779,
        18.07106396791092,
        21.21163662987926,
        24.35247153074930,
        27.49347913204025,
        30.63460646843198,
        33.77582021357357,
        36.91709835366404,
        40.05842576462824,
        43.19979171317673,
        46.34118837166181,
        49.48260989739782,
        52.62405184111500,
        55.76551075501998,
        58.90698392608094,
        62.04846919022717,
        65.18996480020686,
        68.33146932985680,
        71.47298160359373,
        74.61450064370184,
        77.75602563038806,
        80.89755587113763,
        84.03909077693819,
        87.18062984364115,
        90.32217263721048,
        93.46371878194477,
        96.60526795099627,
        99.74681985868060,
        102.8883742541948,
        106.0299309164516,
        109.1714896498054,
        112.3130502804949,
        115.4546126536669,
        118.5961766308725,
        121.7377420879510,
        124.8793089132329,
        128.0208770060083,
        131.1624462752139,
        134.3040166383055,
        137.4455880202843,
        140.5871603528543,
        143.7287335736897,
        146.8703076257966,
        150.0118824569548,
        153.1534580192279,
        156.2950342685335,
        159.4366111642631,
        162.5781886689467,
        165.7197667479550,
        168.8613453692358,
        172.0029245030782,
        175.1445041219027,
        178.2860842000738,
        181.4276647137311,
        184.5692456406387,
        187.7108269600494,
        190.8524086525815,
        193.9939907001091,
        197.1355730856614,
        200.2771557933324,
        203.4187388081986,
        206.5603221162445,
        209.7019057042941,
        212.8434895599495,
        215.9850736715340,
        219.1266580280406,
        222.2682426190843,
        225.4098274348593,
        228.5514124660988,
        231.6929977040385,
        234.8345831403832,
        237.9761687672757,
        241.1177545772680,
        244.2593405632957,
        247.4009267186528,
        250.5425130369700,
        253.6840995121931,
        256.8256861385644,
        259.9672729106045,
        263.1088598230955,
        266.2504468710659,
        269.3920340497761,
        272.5336213547049,
        275.6752087815375,
        278.8167963261531,
        281.9583839846149,
        285.0999717531596,
        288.2415596281877,
        291.3831476062552,
        294.5247356840650,
        297.6663238584589,
        300.8079121264111,
        303.9495004850206,
        307.0910889315050,
        310.2326774631950,
        313.3742660775278
    },
    {   // m = 1
        3.831705970207512,
        7.015586669815613,
        10.17346813506272,
        13.32369193631422,
        16.47063005087763,
        19.61585851046824,
        22.76008438059277,
        25.90367208761838,
        29.04682853491686,
        32.18967991097440,
        35.33230755008387,
        38.47476623477162,
        41.61709421281445,
        44.75931899765282,
        47.90146088718545,
        51.04353518357151,
        54.18555364106132,
        57.32752543790101,
        60.46945784534749,
        63.61135669848123,
        66.75322673409849,
        69.89507183749577,
        73.03689522557383,
        76.17869958464146,
        79.32048717547630,
        82.46225991437356,
        85.60401943635023,
        88.74576714492631,
        91.88750425169499,
        95.02923180804470,
        98.17095073079078,
        101.3126618230387,
        104.4543657912828,
        107.5960632595092,
        110.7377547808992,
        113.8794408475950,
        117.0211218988924,
        120.1627983281490,
        123.3044704886357,
        126.4461386985166,
        129.5878032451040,
        132.7294643885096,
        135.8711223647890,
        139.0127773886597,
        142.1544296558590,
        145.2960793451959,
        148.4377266203422,
        151.5793716314014,
        154.7210145162860,
        157.8626554019303,
        161.0042944053620,
        164.1459316346496,
        167.2875671897441,
        170.4292011632266,
        173.5708336409759,
        176.7124647027638,
        179.8540944227884,
        182.9957228701530,
        186.1373501092955,
        189.2789762003760,
        192.4206011996257,
        195.5622251596626,
        198.7038481297771,
        201.8454701561909,
        204.9870912822923,
        208.1287115488501,
        211.2703309942078,
        214.4119496544620,
        217.5535675636242,
        220.6951847537694,
        223.8368012551717,
        226.9784170964295,
        230.1200323045791,
        233.2616469052006,
        236.4032609225143,
        239.5448743794699,
        242.6864872978287,
        245.8280996982398,
        248.9697116003099,
        252.1113230226686,
        255.2529339830281,
        258.3945444982395,
        261.5361545843441,
        264.6777642566215,
        267.8193735296346,
        270.9609824172707,
        274.1025909327807,
        277.2441990888146,
        280.3858068974556,
        283.5274143702514,
        286.6690215182434,
        289.8106283519944,
        292.9522348816139,
        296.0938411167825,
        299.2354470667741,
        302.3770527404775,
        305.5186581464156,
        308.6602632927644,
        311.8018681873705,
        314.9434728377672
    },
    {   // m = 2
        5.135622301840683,
        8.417244140399855,
        11.61984117214906,
        14.79595178235126,
        17.95981949498783,
        21.11699705302185,
        24.27011231357310,
        27.42057354998456,
        30.56920449551640,
        33.71651950922270,
        36.86285651128381,
        40.00844673347819,
        43.15345377837146,
        46.29799667723692,
        49.44216411041687,
        52.58602350681596,
        55.72962705320114,
        58.87301577261216,
        62.01622235921765,
        65.15927319075780,
        68.30218978418346,
        71.44498986635785,
        74.58768817360240,
        77.73029705697890,
        80.87282694624476,
        84.01528670954617,
        87.15768393520335,
        90.30002515459292,
        93.44231602001113,
        96.58456144778320,
        99.72676573429280,
        102.8689326507279,
        106.0110655209634,
        109.1531672859820,
        112.2952405574717,
        115.4372876626644,
        118.5793106820416,
        121.7213114811962,
        124.8632917378812,
        128.0052529650732,
        131.1471965307178,
        134.2891236747031,
        137.4310355235027,
        140.5729331028549,
        143.7148173487775,
        146.8566891171685,
        149.9985491921996,
        153.1403982936759,
        156.2822370835081,
        159.4240661714182,
        162.5658861199848,
        165.7076974491122,
        168.8495006400029,
        171.9912961386923,
        175.1330843592040,
        178.2748656863716,
        181.4166404783663,
        184.5584090689669,
        187.7001717696018,
        190.8419288711887,
        193.9836806457961,
        197.1254273481458,
        200.2671692169733,
        203.4089064762631,
        206.5506393363703,
        209.6923679950420,
        212.8340926383481,
        215.9758134415304,
        219.1175305697792,
        222.2592441789439,
        225.4009544161840,
        228.5426614205660,
        231.6843653236132,
        234.8260662498091,
        237.9677643170635,
        241.1094596371395,
        244.2511523160499,
        247.3928424544218,
        250.5345301478347,
        253.6762154871325,
        256.8178985587138,
        259.9595794448002,
        263.1012582236860,
        266.2429349699696,
        269.3846097547690,
        272.5262826459230,
        275.6679537081773,
        278.8096230033594,
        281.9512905905409,
        285.0929565261893,
        288.2346208643102,
        291.3762836565799,
        294.5179449524694,
        297.6596047993613,
        300.8012632426584,
        303.9429203258862,
        307.0845760907891,
        310.2262305774202,
        313.3678838242265,
        316.5095358681284
    },
    {   // m = 3
        6.380161895923984,
        9.761023129981667,
        13.01520072169843,
        16.22346616031877,
        19.40941522643501,
        22.58272959310444,
        25.74816669929498,
        28.90835078092176,
        32.06485240709771,
        35.21867073861011,
        38.37047243475694,
        41.52071967040678,
        44.66974311661725,
        47.81778569153330,
        50.96502990620518,
        54.11161556982187,
        57.25765160449901,
        60.40322413847212,
        63.54840217856721,
        66.69324166737268,
        69.83778843790434,
        72.98208040043201,
        76.12614918477410,
        79.27002139005586,
        82.41371954726788,
        85.55726286883000,
        88.70066783822206,
        91.84394867814709,
        94.98711772546561,
        98.13018573387489,
        101.2731621200798,
        104.4160551653968,
        107.5588721819325,
        110.7016196503949,
        113.8443033350319,
        116.9869283800093,
        120.1294993906337,
        123.2720205021307,
        126.4144954381477,
        129.5569275607296,
        132.6993199131840,
        135.8416752569877,
        138.9839961036805,
        142.1262847425225,
        145.2685432645577,
        148.4107735836172,
        151.5529774547062,
        154.6951564901485,
        157.8373121738001,
        160.9794458735966,
        164.1215588526579,
        167.2636522791409,
        170.4057272349997,
        173.5477847237951,
        176.6898256776687,
        179.8318509635862,
        182.9738613889370,
        186.1158577065655,
        189.2578406193015,
        192.3998107840462,
        195.5417688154639,
        198.6837152893237,
        201.8256507455279,
        204.9675756908628,
        208.1094906014995,
        211.2513959252716,
        214.3932920837524,
        217.5351794741522,
        220.6770584710535,
        223.8189294279994,
        226.9607926789509,
        230.1026485396240,
        233.2444973087189,
        236.3863392690516,
        239.5281746885954,
        242.6700038214426,
        245.8118269086917,
        248.9536441792675,
        252.0954558506799,
        255.2372621297268,
        258.3790632131455,
        261.5208592882173,
        264.6626505333286,
        267.8044371184931,
        270.9462192058375,
        274.0879969500530,
        277.2297704988175,
        280.3715399931881,
        283.5133055679684,
        286.6550673520509,
        289.7968254687379,
        292.9385800360411,
        296.0803311669622,
        299.2220789697565,
        302.3638235481789,
        305.5055650017159,
        308.6473034258026,
        311.7890389120269,
        314.9307715483214,
        318.0725014191441
    },
    {   // m = 4
        7.588342434503804,
        11.06470948850119,
        14.37253667161759,
        17.61596604980483,
        20.82693295696239,
        24.01901952477111,
        27.19908776598125,
        30.37100766711725,
        33.53713771181922,
        36.69900112874465,
        39.85762730218089,
        43.01373772335443,
        46.16785351292437,
        49.32036068639027,
        52.47155139845802,
        55.62165090976798,
        58.77083574045925,
        61.91924620409770,
        65.06699525569557,
        68.21417486146705,
        71.36086066529799,
        74.50711546139641,
        77.65299181534341,
        80.79853406792371,
        83.94377988509808,
        87.08876146981360,
        90.23350651879237,
        93.37803898484892,
        96.52237968938122,
        99.66654681832882,
        102.8105563266903,
        105.9544222706969,
        109.0981570823107,
        112.2417717974093,
        115.3852762465297,
        118.5286792151526,
        121.6719885790571,
        124.8152114191589,
        127.9583541193705,
        131.1014224503452,
        134.2444216414238,
        137.3873564426807,
        140.5302311786232,
        143.6730497948259,
        146.8158158985625,
        149.9585327943162,
        153.1012035149084,
        156.2438308488629,
        159.3864173645263,
        162.5289654313860,
        165.6714772389565,
        168.8139548135532,
        171.9564000332244,
        175.0988146410723,
        178.2412002571628,
        181.3835583891960,
        184.5258904420842,
        187.6681977265669,
        190.8104814669731,
        193.9527428082286,
        197.0949828221911,
        200.2372025133895,
        203.3794028242294,
        206.5215846397236,
        209.6637487917961,
        212.8058960632053,
        215.9480271911235,
        219.0901428704084,
        222.2322437565969,
        225.3743304686482,
        228.5164035914600,
        231.6584636781807,
        234.8005112523345,
        237.9425468097791,
        241.0845708205096,
        244.2265837303231,
        247.3685859623567,
        250.5105779185092,
        253.6525599807571,
        256.7945325123746,
        259.9364958590641,
        263.0784503500062,
        266.2203962988352,
        269.3623340045463,
        272.5042637523399,
        275.6461858144082,
        278.7881004506686,
        281.9300079094478,
        285.0719084281212,
        288.2138022337098,
        291.3556895434388,
        294.4975705652603,
        297.6394454983431,
        300.7813145335317,
        303.9231778537763,
        307.0650356345373,
        310.2068880441645,
        313.3487352442539,
        316.4905773899829,
        319.6324146304257
    }
};

}}} // namespace plask::optical::slab
