# install pre-requisites. tested on ubuntu 16.04 (14.04 should mostly work except for installing ode part, which may
# require manual installation

echo "Install pre-requisites"

apt-get update

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    if [[ `lsb_release -rs` == "14.04" ]]; then
        sudo apt-get install -y libav-tools
    else
        sudo apt-get install -y ffmpeg
    fi
else
    echo "OS version not tested yet"
fi

apt-get install -y\
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
    python3-dev  \
    libode-dev \
    libopenmpi-dev


sudo apt-get install -y build-essential cmake pkg-config git ffmpeg
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
pip3 install tensorflow

echo "Start Dart Installation"

# install dart
git clone https://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.3.0
cp ../external/lcp.cpp dart/external/odelcpsolver/lcp.cpp
mkdir build && cd build
cmake ..
make -j4
sudo make install

cd ../../

echo "Start Pydart2 Installation"

# install pydart
git clone https://github.com/sehoonha/pydart2.git
cp external/pydart2_draw.cpp pydart2/pydart2/
cd pydart2
python3 setup.py build build_ext
sudo python3 setup.py develop

echo "Start Baselines Installation"

cd ../baselines
sudo pip3 install -e .

echo "Start DartEnv Installation"

cd ../dart-env
sudo pip3 uninstall -y gym
sudo pip3 install -e .

sudo pip3 install mpi4py
sudo pip3 install matplotlib

cd ..

echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$PWD:$PYTHONPATH' >> ~/.bashrc

echo "Installation Done"