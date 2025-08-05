# Phils model

This is a fully connected neural network. It can solve regression and classification problems. It also supports Adam optimizer. There are code sections where I used processor-specific functions (for x86 and ARM processors) for better optimization of the program.

To run the `main.ipynb` file, use this:

```sh
brew install llvm

clang -shared -o main.so -fPIC -O3 -fopenmp -ffast-math -march=native main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c src/forward.c src/backward.c src/logger.c src/predict.c
```

> You may donate to [phils_model](https://github.com/PHILIPP111007/ML/tree/main/phils_model) project:
>
> * Ethereum: 0xE2e2D675a3843f4ED211BB93847ad18b0A6fe7c6
