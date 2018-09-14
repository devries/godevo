package main

import (
	"bitbucket.org/devries/godevo"
	"fmt"
	"gonum.org/v1/gonum/stat"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	x := make([]float64, 21)
	p0 := []float64{2.0, 3.0}

	for i := range x {
		x[i] = 0.5 * float64(i)
	}

	y := make([]float64, 21)

	for i := range y {
		y[i] = linear(p0, x[i]) + rand.NormFloat64()*0.5
	}

	var weights []float64
	alpha, beta := stat.LinearRegression(x, y, weights, false)
	fmt.Printf("slope: %f, intercept: %f\n", beta, alpha)

	optimizationFunc := func(params []float64) float64 {
		sumsq := 0.0
		for i := range x {
			sumsq += residual(params, x[i], y[i], 0.25)
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

	for i := 0; i < 50000; i++ {
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
