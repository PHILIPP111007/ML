# Phils model

To run the `main_with_c.ipynb` file, use this:

```
clang -shared -o main.so -fPIC -O3 main.c src/functions.c src/activations.c src/loss.c src/init.c src/json.c src/adam.c
```
