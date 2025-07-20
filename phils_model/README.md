# Phils model

To run the `main.ipynb` file, use this:

```sh
brew install llvm

clang -shared -o main.so -fPIC -O3 main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c src/forward.c src/backward.c src/get_time.c
```

You may donate to ML project:

Ethereum: 0xE2e2D675a3843f4ED211BB93847ad18b0A6fe7c6
