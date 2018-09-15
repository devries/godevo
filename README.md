# Differential Evolution in Go

## Introduction

The differential evolution algorithm is an optimization algorithm developed by
[Storn & Price
(1997)](https://link.springer.com/article/10.1023%2FA%3A1008202821328). Given
a function with a set of `N` parameters, the algorithm creates a set of `NP`
random solutions within a defined parameter space, and then uses the variation
among the parameters to mutate the existing population and find better
solutions. Although this optimization method can be slow, it can be useful to
avoid the falling into a local minimum, and it is a simple algorithm to
implement and understand.

Differential Evolution can also be adapted to Markov Chain Monte Carlo methods
easily (see [ter Braak
2006](https://link.springer.com/article/10.1007%2Fs11222-006-8769-1)). This
package implements differential evolution and Markov chain Monte Carlo
differential evolution in the Go programming language.

## Using the library

Import the library with a standard import statement.

```go
import "github.com/devries/godevo"
```

Next, you need a function which takes a set of parameters as a go slice of
`float64` numbers, and returns a value to be optimized as a `float64`. For
example, the function below is a parabola in two dimensions which takes a
slice of length 2 as input and returns the parabola value.

```go
func parabola(vec []float64) float64 {
        res := 3.0 + (vec[0]-1.0)*(vec[0]-1.0) + (vec[1]-2.0)*(vec[1]-2.0)

        return res
}
```

The minimization process involves specifying the range of parameters available
by creating a set of minimum and maximum parameter values. In the code below I
allow the first parameter to vary between 0.0 and 2.0, and the second
parameter to vary between 0.0 and 5.0. I choose `NP` which is the size of the
population of samples to be 15 in the case below. I indicate if I want the
optimization function to be calculated in parallel (if the function takes a
long time to calculate this can be a good idea, but if it is relatively quick
to calculate, then the overhead of goroutines can slow the process down).
Finally, I provide the function to be optimized into the `Initialize`
function. I get a pointer to a `Model` as well as an `error` which will be
`nil` if the initialization worked.

```go
minparams := []float64{0.0, 0.0}
maxparams := []float64{2.0, 5.0}

model, err := godevo.Initialize(minparams, maxparams, 15, false, parabola)
if err != nil {
        panic(err)
}
```

It is possible to fine tune parameters after initialization, for example the
crossover constant `CR` and the weighting factor `F` can be modified on the
model structure. By default `CR=0.1` and `F=0.7`.

```go
model.CrossoverConstant = 0.4
model.WeightingFactor = 0.8
```

The method `Step` of the receiver `*Model` performs one iteration (one
generation) of the algorithm. Therefore to run through 50 generations of
evolution, you could do the following:

```go
for i := 0; i < 50; i++ {
        model.Step()
}
```

At any point you can use the `Best` method of `*Model` to get the best
solution and the fitness of that solution.

```go
bestParameters, bestFitness := model.Best()
```

## Markov Chain Monte Carlo

The Markov chain Monte Carlo method is very similar, but is initialized with
the `InitializeMCMC` function, which takes the same parameter, but sets the
optimization function to accept solutions based on the Metropolis algorithm,
and only evolves child generations directly from the corresponding parent
solution, which is required to get good statistical measurements. Generations
are evolved using the `Step` method, but generally you want to find the mean
value of the parameters and their standard deviations in this case. For that
you can use the `MeanStd` method of the `*Model` receiver as shown below.

```go
meanParameters, stdevParameters := model.MeanStd()
```

The Markov chain Monte Carlo variant should provide a population whose
distribution is a good statistical representation of the accuracy of the
model.

## Documentation

Detailed documentation is available [through
godoc](https://godoc.org/github.com/devries/godevo). Some examples are
included in the `internal` subdirectory.
