package godevo_test

import (
	"fmt"
	"math/rand"

	"github.com/devries/godevo"
)

func Example() {
	rand.Seed(42)
	minparams := []float64{0.0, 0.0}
	maxparams := []float64{2.0, 5.0}

	model, err := godevo.Initialize(minparams, maxparams, 15, false, parabola)
	if err != nil {
		panic(err)
	}
	model.TrialFunction = godevo.TrialPopulationSP95
	model.CrossoverConstant = 0.4
	model.WeightingFactor = 0.8
	model.ParallelMode = false

	bp, bf := model.Best()
	fmt.Printf("Best Parameters: %v\n", bp)
	fmt.Printf("Best Fitness: %f\n", bf)

	ngen := 50
	fmt.Printf("Running model of %d generations\n", ngen)

	for i := 0; i < ngen; i++ {
		model.Step()
	}

	bp, bf = model.Best()
	fmt.Printf("Best Parameters: %v\n", bp)
	fmt.Printf("Best Fitness: %f\n", bf)
	// Output:
	// Best Parameters: [1.3280218953690182 2.331018106387252]
	// Best Fitness: 3.217171
	// Running model of 50 generations
	// Best Parameters: [0.999987025007789 1.999966608737686]
	// Best Fitness: 3.000000
}

func parabola(vec []float64) float64 {
	res := 3.0 + (vec[0]-1.0)*(vec[0]-1.0) + (vec[1]-2.0)*(vec[1]-2.0)

	return res
}

func Example_mcmc() {
	rand.Seed(42)
	x := make([]float64, 21)
	p0 := []float64{2.0, 3.0}

	for i := range x {
		x[i] = 0.5 * float64(i)
	}

	y := make([]float64, 21)

	for i := range y {
		y[i] = linear(p0, x[i]) + rand.NormFloat64()*0.5
	}

	optimizationFunc := func(params []float64) float64 {
		sumsq := 0.0
		for i := range x {
			sumsq += residual(params, x[i], y[i], 0.25)
		}

		return sumsq
	}

	pmin := []float64{0.0, 0.0}
	pmax := []float64{10.0, 10.0}

	// This is not actually efficient in parallel, but I just want to exercise that code.
	model, err := godevo.InitializeMCMC(pmin, pmax, 500, true, optimizationFunc)
	if err != nil {
		panic(err)
	}

	model.WeightingFactor = 0.9

	for i := 0; i < 2000; i++ {
		model.Step()
	}

	mean, stdev := model.MeanStd()
	fmt.Printf("Mean Result: %v\n", mean)
	fmt.Printf("Standard Deviation: %v\n", stdev)
	// Output:
	// Mean Result: [1.9624802929430822 3.3155542023129514]
	// Standard Deviation: [0.03711446105839528 0.21234935229456098]

}

func linear(params []float64, x float64) float64 {
	return params[0]*x + params[1]
}

func residual(params []float64, x, y, sigmasq float64) float64 {
	yp := linear(params, x)

	diff := yp - y

	return diff * diff / sigmasq
}
