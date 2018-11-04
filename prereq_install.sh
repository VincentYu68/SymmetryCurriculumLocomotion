
echo "Install pre-requisites"

apt-get update \
    && apt-get install -y libav-tools \
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    python-opengl \
    libboost-all-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \
    xpra \
    libav-tools  \
    python3-dev  \
    libode-dev

apt-add-repository ppa:libccd-debs
apt-add-repository ppa:fcl-debs
apt-add-repository ppa:dartsim
apt-get update
apt-get install swig
apt-get install swig python-pip python-qt4 python-qt4-dev python-qt4-gl
apt-get install python3-pip python3-pyqt4 python3-pyqt4.qtopengl

echo "Start Dart Installation"

# install dart
git clone https://github.com/dartsim/dart.git
git checkout tags/v6.4.0
cd dart
mkdir build && cd build
cmake ..
make -j4
sudo make install

cd ../../

echo "Start Pydart2 Installation"

# install pydart
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python3 setup.py build build_ext
sudo python3 setup.py develop

export PYTHONPATH=$PWD:$PYTHONPATH

echo "Start Baselines Installation"

cd baselines
pip install -e .

echo "Start DartEnv Installation"

cd ../dart-env
pip install -e .

cd ..

echo "Installation Done"