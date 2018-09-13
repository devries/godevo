package main

import (
	"bitbucket.org/devries/godevo"
	"fmt"
	"math/rand"
)

func main() {
	x := make([]float64, 21)
	p0 := []float64{2.0, 3.0}

	for i := range x {
		x[i] = 0.5 * float64(i)
	}

	y := make([]float64, 21)

	for i := range y {
		y[i] = linear(p0, x[i]) + rand.NormFloat64()*2.0
	}

	optimizationFunc := func(params []float64) float64 {
		sumsq := 0.0
		for i := range x {
			sumsq += residual(params, x[i], y[i], 4.0)
		}

		return sumsq
	}

	pmin := []float64{0.0, 0.0}
	pmax := []float64{10.0, 10.0}

	model, err := godevo.InitializeMCMC(pmin, pmax, 500, optimizationFunc)
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
}

func linear(params []float64, x float64) float64 {
	return params[0]*x + params[1]
}

func residual(params []float64, x, y, sigmasq float64) float64 {
	yp := linear(params, x)

	diff := yp - y

	return diff * diff / sigmasq
}
