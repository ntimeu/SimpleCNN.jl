# SimpleCNN.jl

SimpleCNN.jl is an example Convolutional Neural Network showcasing Julia and
the [Flux.jl](https://fluxml.ai/) library.

## How to install it

From Julia's REPL :

```julia
julia> ]
pkg> add https://github.com/ntimeu/SimpleCNN.jl#0.1.0
```

## How to use it

Once installed, the following commands will import and start training a simple
CNN network on the FashionMNIST dataset.

```julia
import SimpleCNN: main

main()
```
