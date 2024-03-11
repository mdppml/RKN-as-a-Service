# PPAURORA

Document the problems to think about at a later state here. If particular file(s) are affected by the problem, mention those at the beginning. The style for problem documentation is:
##### <affected file(s)>
-> problem description

### Computation Specification
##### Programm start
-> define a buffer size depending on machine the program is running on
##### Files processing vectors (multiple elements)
-> what is the maximum number of elements a matrix can contain without getting overflow problems during computations?


### Structure
##### proxy.cpp in apps/test
-> Should we include different folders for test of different applications or include all apps_tests there?
##### utils
-> Creating separate files for addVal2..., convert... and other functions in flib.h ?


### Installation for ONNX file reading: 
###### see https://github.com/protocolbuffers/protobuf/blob/main/src/README.md
sudo apt-get install autoconf automake libtool curl make g++ unzip
sudo apt-get install python3-pip python3-dev libprotobuf-dev protobuf-compiler
###### see https://github.com/onnx/onnx
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
#### prefer lite proto
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e .

#### if it does not work: 
pip install onnx


### see https://github.com/leimao/ONNX-Runtime-Inference (did not work for me)