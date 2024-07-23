# RKN-as-a-Service

RKN-as-a-Service is a cloud-based three-party computational privacy preserving protein fold recognition approach. [The paper][https://www.cell.com/patterns/fulltext/S2666-3899(24)00156-9] is published in Patterns. RKN-as-a-service protein fold recognition is implemented in C++.

## Installation

No installation is required.

Make sure to clone the repository using the "--recurse-submodules" or "--recurse" flag to initialise the submodules as well.

If you already have a local version of this repository without submodules, use the command "git submodule update --init --recursive" to initialise the submodules.

The benchmark bash script requires [toxiproxy](https://github.com/Shopify/toxiproxy/releases/latest). Drop the toxiproxy-server and toxiproxy-cli executables into the same directory as the script.

## Compiling

### Building with CMake
After cloning the repo into directory `RKN-as-a-Service`, you can build the library `RKN-as-a-Service` by executing the following commands. 
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
After the build completes, the output binaries can be found in `RKN-as-a-Service/build/` directory 

## Usage

The following bash script describes the template of how privacy preserving inference on a pre-trained RKN can be called:

```bash
./helper <ip of helper> <port of helper>
./proxy_rkn Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <random flag> <number of anchor points> <length of kmers> <lambda> <sigma> <run id> <network> <test sample index>
./proxy_rkn Role <port of proxy 1> <ip of proxy 1> <port of helper> <ip of helper> <random flag> <number of anchor points> <length of kmers> <lambda> <sigma> <run id> <network> <test sample index>
```

Example bash scripts are as follows:
```bash
./helper_rkn 127.0.0.1 7777
./proxy_rkn  0 9999 "127.0.0.1" 7777 "127.0.0.1" 0 128 10 0.5 0.4 1 lan 12
./proxy_rkn  1 9999 "127.0.0.1" 7777 "127.0.0.1" 0 128 10 0.5 0.4 1 lan 12
```

It is also possible to run the experiments through bash scripts in `exp_runners/rkn_experiments`. In this folder, _real dataset_ experiments can be run as follows:
```bash
./real_data_experiments.sh
```
It will iterate over the specified test samples with specified amount of repetitions. The results will be saved in `exp_runners/rkn_experiments/pprkn_inference_results/real` folder. Similarly, the synthetic data experiments can be run using the following bash script:
```bash
./synthetic_data_experiments.sh
```
The results will be saved in `exp_runners/rkn_experiments/pprkn_inference_results/synthetic` folder.


## License
[MIT](https://choosealicense.com/licenses/mit/)
