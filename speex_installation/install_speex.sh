#!/bin/bash
cd $HOME
if [[ ! -e project ]]; then
	    mkdir project
    fi
    cd project
    git clone https://github.com/xiph/speex.git   
    cd speex
    sudo apt-get install autoconf
    sudo apt-get install libtool
    sudo apt install make
    sudo apt-get install build-essential libspeex-dev libspeexdsp-dev libpulse-dev
    bash autogen.sh
    bash configure --prefix=$HOME
    make
    sudo make install
