# install pre-requisites. tested on ubuntu 16.04 (14.04 should mostly work except for installing ode part, which may
# require manual installation

echo "Install pre-requisites"

apt-get update \
    && apt-get install -y libav-tools \
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

sudo apt-get install -y build-essential cmake pkg-config git
sudo apt-get install -y libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install -y libopenscenegraph-dev
sudo apt-get install -y libbullet-dev
sudo apt-get install -y liburdfdom-dev
sudo apt-get install -y libnlopt-dev
sudo apt-get install -y libxi-dev libxmu-dev freeglut3-dev
sudo apt-get install -y libtinyxml2-dev

apt-get install -y swig
apt-get install -y swig python-pip python-qt4 python-qt4-dev python-qt4-gl
apt-get install -y python3-pip python3-pyqt4 python3-pyqt4.qtopengl

pip3 install numpy

echo "Start Dart Installation"

# install dart
git clone https://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.3.0
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
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

echo "Start Baselines Installation"

cd baselines
pip3 install -e .

echo "Start DartEnv Installation"

cd ../dart-env
pip3 install -e .

cd ..

echo "Installation Done"