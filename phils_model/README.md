# Phils model

This is a fully connected neural network. It can solve regression and classification problems. It also supports Adam optimizer. There are code sections where I used processor-specific instructions (for x86 and ARM processors) for better optimization of the program. OpenCL GPU computing support has recently been added.

To run the `main.ipynb` file, use this:

For MacOS:

```sh
sudo xcode-select --install
brew install llvm libomp
brew install opencl-headers
brew doctor

clang -shared \
    -o main.so \
    -fPIC \
    -O3 \
    -fopenmp \
    -ffast-math \
    -march=native \
    main.c src/linear.c src/src/functions.c src/src/activations.c src/src/loss.c src/src/init.c src/src/json.c src/src/adam.c src/src/forward.c src/src/backward.c src/src/logger.c src/src/predict.c \
    -framework OpenCL \
    -I/opt/homebrew/opt/opencl-headers/include
```

For Linux:

```sh
sudo apt-get update
sudo apt-get install ocl-icd-opencl-dev
sudo apt install nvidia-cuda-toolkit

gcc -shared \
    -o main.so \
    -fPIC \
    -fno-wrapv \
    -O3 \
    -fopenmp \
    -march=native \
    main.c src/linear.c src/src/functions.c src/src/activations.c src/src/loss.c src/src/init.c src/src/json.c src/src/adam.c src/src/forward.c src/src/backward.c src/src/logger.c src/src/predict.c \
    -lOpenCL

# If you have OpenCl version lower than 3.0

conda install cuda-opencl cuda-opencl-dev ocl-icd-system

gcc -shared \
    -o main.so \
    -fPIC \
    -fno-wrapv \
    -O3 \
    -fopenmp \
    -march=native \
    main.c src/linear.c src/src/functions.c src/src/activations.c src/src/loss.c src/src/init.c src/src/json.c src/src/adam.c src/src/forward.c src/src/backward.c src/src/logger.c src/src/predict.c \
    -I/home/phil/micromamba/envs/phils_model/include \
    -lOpenCL
```
