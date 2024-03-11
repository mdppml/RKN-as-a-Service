# CECILIA

CECILIA is a three-party computational framework that offers a variety of building blocks to facilitate more complex algorithms in a privacy preserving manner. It is implemented in C++.
## Links to Related Papers

[CECILIA: Comprehensive Secure Machine Learning Framework](https://arxiv.org/abs/2202.03023)

[ppAURORA: Privacy Preserving Area Under Receiver Operating Characteristic and Precision-Recall Curves](https://arxiv.org/abs/2102.08788)

## Installation

No installation is required.

Make sure to clone the repository using the "--recurse-submodules" or "--recurse" flag to initialise the submodules as well.

If you already have a local version of this repository without submodules, use the command "git submodule update --init --recursive" to initialise the submodules.

The benchmark bash script requires [toxiproxy](https://github.com/Shopify/toxiproxy/releases/latest). Drop the toxiproxy-server and toxiproxy-cli executables into the same directory as the script.

## Compiling

### Building with CMake
After cloning the repo into directory `CECILIA`, you can build the library `CECILIA` by executing the following commands. 
```bash
mkdir build
cd build
```

```bash
cmake -S ../ -G -DCMAKE_BUILD_TYPE=Release
```

```bash
make
```
After the build completes, the output binaries can be found in `CECILIA/build/` directory 
### Building with C++
### AUROC

#### helper

```bash
c++ -std=gnu++17 -pthread -W -O3 apps/proxy.cpp core/Party.cpp core/Party.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h examples/auroc/llib.h -o proxy_auroc
```

#### proxy

```bash
c++ -std=gnu++17 -pthread -W -O3 apps/helper.cpp core/Party.cpp core/Party.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h -o helper_auroc
```

### AUROC WITH TIE

#### helper

```bash
c++ -std=gnu++17 -pthread -W -O3 examples/auroctie/proxy.cpp core/Party.cpp core/Party.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h examples/auroctie/llib.h -o proxy_auroctie
```

#### proxy

```bash
c++ -std=gnu++17 -pthread -W -O3 apps/helper.cpp core/Party.cpp core/Party.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h -o helper_auroctie
```

### AUPRC

#### helper

```bash
c++ -std=gnu++17 -pthread -W -O3 examples/aupr/proxy.cpp core/Party.cpp core/Party.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h examples/aupr/llib.h -o proxy_aupr
```

#### proxy

```bash
c++ -std=gnu++17 -pthread -W -O3 apps/helper.cpp core/Party.cpp core/Party.h utils/constant.h utils/parse_options.cpp utils/parse_options.h utils/connection.h utils/flib.h -o helper_aupr
```

## Usage

```bash
./helper_auroc <ip of helper> <port of helper>
./proxy_auroc Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <delta> <input>
./proxy_auroc Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <delta> <input>
```

- input = #input parties,#samples of the first input party,#samples of the second input party,...,#samples of the last input party
- delta = delta is a number that specifies how many selections are made after shuffling

```bash
./helper_auroc "172.31.43.235" 7777
./proxy_auroc 0 8888 "127.0.0.1" 7777 "127.0.0.1" 10 "8,1000,1000,1000,1000,1000,1000,1000,1000"
./proxy_auroc 1 8888 "127.0.0.1" 7777 "127.0.0.1" 10 "8,1000,1000,1000,1000,1000,1000,1000,1000"
```


## CNN
```bash
./helper_cnn <ip of helper> <port of helper> <model>
./proxy_cnn Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <model> 
./proxy_cnn Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <model> 
```
- model = model to be used: [0, 1, 2] with 0 - Chameleon, 1 - MiniONN (MiniONNs model parameter are not consistent), 2 - LeNet5, other value or none - a 4-layer CNN with random weights

```bash
./helper_cnn 127.0.0.1 7777 0
./proxy_cnn 0 8888 127.0.0.1 7777 127.0.0.1 0
./proxy_cnn 1 8888 127.0.0.1 7777 127.0.0.1 0
```

## RKN

Privacy preserving inference on a pre-trained RKN

```bash
./helper <ip of helper> <port of helper>
./proxy_rkn Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <random flag> <number of anchor points> <length of kmers> <lambda> <sigma> 
./proxy_rkn Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <random flag> <number of anchor points> <length of kmers> <lambda> <sigma> 
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
