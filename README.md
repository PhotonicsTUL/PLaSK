# PLaSK

Photonic Laser Simulation Kit is a comprehensive tool for numerical analysis
of broad range of physical phenomena in photonic devices. It has been designed
for simulating mainly semiconductor lasers, however the range of the covered
devices is much larger and includes e.g. transistors, light emitting diodes,
photodetectors, etc. Due to its modular nature it is possible to perform
computations of virtually any physical phenomenon in micro-scale structures.

PLaSK has been originally developed in a Photonics Group of Lodz University of Technology, which
has many-year experience in numerical analysis of semiconductor lasers. Such structures, due to their
complex structure, often fail to be correctly simulated by popular general-purpose software.
However, in designing PLaSK we have taken special care to consider all the special cases
of semiconductor lasers and to choose (or invent where necessary) appropriate algorithms.


## Features

- Multi-physics analysis of photonic devices
- Automatic handling of relations between models
- Advanced modeling of heat flow,current spreading, optical gain, electromagnetic radiation
- Graphical user interface


## Recognition

PLaSK is an effect of many years of scientific work of the members of the Photonics Group at TUL.
We provide it to the world in hope it will be useful. However, if you use it for your work, we would
appreciate if you gave us a proper recognition by citing the following works:

1. M. Dems, P. Beling, M. Gębski, Ł. Piskorski, J. Walczak, M. Kuc, L. Frasunkiewicz, W. Michał, R. Sarzała,
   T. Czyszanowski: VCSEL modeling with self-consistent models: From simple approximations to comprehensive
   numerical analysis, *Proc. SPIE* **9381**, 93810K (2015).
   <https://doi.org/10.1117/12.2078321>

2. R. Sarzała, T. Czyszanowski, M. Wasiak, M. Dems, L. Piskorski, W. Nakwaski, K. Panajotov:
   Numerical self-consistent analysis of VCSELs, *Adv. Opt. Technol.* **2012**, 689519 (2012).
   <https://doi.org/10.1155/2012/689519>

3. Ł. Piskorski, R.P. Sarzała, W. Nakwaski: Self-consistent model of 650 nm GaInP/AlGaInP quantum-well
   vertical-cavity surface-emitting diode lasers, *Semicond. Sci. Technol.* **22**, 593–600 (2007).
   <https://doi.org/10.1088/0268-1242/22/6/002>


4. M. Dems, R. Kotynski, K. Panajotov: Plane Wave Admittance Method — a novel approach for determining
   the electromagnetic modes in photonic structures, *Optics Express* **13**, 3196-3207 (2005).
   <https://doi.org/10.1364/opex.13.003196>


## Building and Installation

### Linux

#### Deb-dased systems (Debian, Ubuntu, Mint, etc.)

Enter the following commands:

    $ sudo apt install g++ git cmake cmake-qt-gui ninja-build libboost-all-dev \
               libeigen3-dev libexpat1-dev libmkl-dev python3-dev \
               python3-numpy python3-scipy python3-matplotlib python3-h5py python3-lxml \
               python3-yaml python3-pyqt5 python3-sphinx python3-pip ipython3 \
               doxygen libgsl-dev libx11-dev qhelpgenerator-qt5 qttools5-dev-tools \
               python3-sphinxcontrib.qthelp

    # On Ubuntu 21.10 the following command is necessary
    $ sudo apt install python3-sphinxcontrib.qthelp

    $ cd _your/chosen/plask/directory_

    $ git clone git@phys.p.lodz.pl:plask.git .

    $ git submodule update --init

    $ mkdir build-release

    $ cd build-release

    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI_TESTING=OFF ..

    $ ninja

If you want to build debug version, replace `release` with `debug` and `Release` with `Debug` in above instruction.

You may also use alternative compiler CLang, which should compile your code faster. To do so, replace the above
`cmake -G Ninja -D CMAKE_BUILD_TYPE=Release ..` command with:

    $ sudo apt install clang

    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..


Additionally, you may install your favorite IDE. E.g. one of the following:

    $ sudo apt install code

    $ sudo apt install qtcreator

    $ sudo apt install codeblocks

    $ sudo apt install kdevelop

In `qtcreator` of `kdevelop` you may open `CMakeList.txt` as a project.

#### Manjaro and Arch Linux

Enter the following commands:

    $ sudo pacman -S --needed gcc cmake git boost boost-libs eigen expat \
           ninja openmp openblas lapack python-numpy python-scipy \
           python-matplotlib python-h5py python-lxml python-yaml \
           pyside2 python-sphinx python-pip ipython doxygen gsl libx11

    $ cd _your/chosen/plask/directory_

    $ git clone git@phys.p.lodz.pl:plask.git .

    $ git submodule update --init

    $ mkdir build-release

    $ cd build-release

    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI_TESTING=OFF ..

    $ ninja

If you want to build debug version, replace `release` with `debug` and `Release` with `Debug` in above instruction.

You may also use alternative compiler CLang, which should compile your code faster. To do so, replace the above
`cmake -G Ninja -D CMAKE_BUILD_TYPE=Release ..` command with:

    $ sudo pacman -S clang

    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..


Additionally, you may install your favorite IDE. E.g. one of the following:

    $ sudo pacman -S code

    $ sudo pacman -S kdevelop

    $ sudo pacman -S qtcreator

    $ sudo pacman -S codeblocks

In `vscode`, `qtcreator` of `kdevelop` you may open `CMakeList.txt` as a project.

### Windows

This is a new instruction for Building PLaSK with Visual Studio and Python 3.

#### Required tools and libraries

- Microsoft Visual Studio 2017, 2019, or 2022 (recommended). **You need to update it to the newest version, or at least 15.9.3!**

- Anaconda Python Distribution 3.9 64-bit. You may install either for the current user (just you) or all users.
  It is recommended to select select ‘Add to Path’ checkbox if possible (if you do not this, you will have to add Anaconda
  to your path manually).

- Additional libraries (see below)...

- Optionally:
  - [Doxygen](https://www.doxygen.nl/download.html#srcbin). Automatic developer documentation builder. It is not necessary.

#### Installing additional libraries

In order to install required libraries (Boost, Expat, and Eigen), please download
[Libraries.zip](https://get.plask.app/packages/Libraries.zip) and extract it into `C:\` folder.
You should see a new folder `C:\Libraries` around 200MB large.

Next you *must* add the folder `C:\Libraries\bin` to the Path environment variable. In Windows 10 and 11, open the Start menu
and start typing `variables`. Then select **Edit environment variables four your account**. In the dialog window that opens,
find **Path** in the **User variables**, select it and click **Edit...**. In the another window, which pop-ups, click **New**,
type **`C:\Libraries\bin`** and finally click **Ok** in all the dialogs.

If you have not selected `Add to Path` while installing Anaconda you must additionally add the following folders to
the PATH (replace `C:\ProgramData\Anaconda3` with the folder you installed Anaconda into): `C:\ProgramData\Anaconda3`,
`C:\ProgramData\Anaconda3\Library\bin`, `C:\ProgramData\Anaconda3\Library\usr\bin`, `C:\ProgramData\Anaconda3\Scripts`.

Eventually your Path environment variable must contain the following folders:

- `C:\ProgramData\Anaconda3`
- `C:\ProgramData\Anaconda3\Library\bin`
- `C:\ProgramData\Anaconda3\Library\usr\bin`
- `C:\ProgramData\Anaconda3\Scripts`
- `C:\Libraries\bin`

Next open an `Anaconda Prompt` as administrator and type two commands:

    conda install mkl-devel


#### Compile PLaSK under Visual Studio

Start Microsoft Visual Studio. Open the file `CMakeLists.txt` from PLaSK source folder, by using
**File** → **Open** → **CMake...** menu. Select **x64-Debug** or **x64-Release** configuration (*x86* build will fail).
Use the menu **CMake** → **Build All** to build PLaSK.

#### Debugging

To set up command line arguments for the debugger, refer to
<https://docs.microsoft.com/en-us/cpp/ide/cmake-tools-for-visual-cpp#configuring-cmake-debugging-sessions>.
We suggest that you use the target `${buildRoot}\bin\plask.exe` for debugging.
