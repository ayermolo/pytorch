#!/bin/bash

set -ex

function install_ucx() {
  set -ex
  sudo apt-get install -y libtool
  git clone --recursive https://github.com/openucx/ucx.git
  pushd ucx
  git checkout ${UCX_COMMIT}
  ./autogen.sh
  ./configure --prefix=$UCX_HOME      \
      --enable-mt                     \
      --enable-profiling              \
      --enable-stats
  time make -j
  sudo make install
  popd
  rm -rf ucx
}

function install_ucc() {
  set -ex
  git clone --recursive https://github.com/openucx/ucc.git
  pushd ucc
  git checkout ${UCC_COMMIT}
  ./autogen.sh
  ./configure --prefix=$UCC_HOME --with-ucx=$UCX_HOME
  time make -j
  sudo make install
  popd
  rm -rf ucc
}

install_ucx
install_ucc
