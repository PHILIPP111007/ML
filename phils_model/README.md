# Phils model

To run the `main.ipynb` file, use this:

```sh
brew install llvm

clang -shared -fopenmp -o main.so -fPIC -O3 main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c

export OMP_NUM_THREADS=7
```
