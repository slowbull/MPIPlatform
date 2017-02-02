# install g++ openmpi
sudo apt-get update
sudo apt-get -y  install libopenmpi-dev
sudo apt-get -y install g++
sudo apt-get -y install openmpi-bin
sudo apt-get -y install make
sudo apt-get -y install cmake

#install openblas
sudo apt-get -y install libopenblas-dev

#install lapack
sudo apt-get -y install liblapack-dev

#install arpack
sudo apt-get -y install libarpack-dev

# install armadillo
if [ ! -f armadillo-7.200.2.tar.xz ]
then
	wget http://sourceforge.net/projects/arma/files/armadillo-7.200.2.tar.xz
fi 
tar xvf armadillo-7.200.2.tar.xz
cd armadillo-7.200.2
cmake .
make 
sudo make install
cd ..
rm armadillo-7.200.2.tar.xz 

#fetch gflags
git clone https://github.com/gflags/gflags.git

mkdir build
cd build
# generate splitdata
g++ -o splitdata ../data/split_data.c

# make project
cmake ..
make

