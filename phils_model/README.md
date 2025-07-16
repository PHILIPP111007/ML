# Phils model

To run the `main_with_c.ipynb` file, use this:

```sh
brew install libomp
```

libomp is keg-only, which means it was not symlinked into /usr/local,
because it can override GCC headers and result in broken builds.

For compilers to find libomp you may need to set:

```sh
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
```

```
clang -fopenmp -shared -o functions.so -fPIC -O3 functions.c
```
