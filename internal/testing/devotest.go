package main

import (
	"bitbucket.org/devries/godevo"
	"fmt"
)

func main() {
	minparams := []float64{0.0, 0.0}
	maxparams := []float64{2.0, 5.0}

	model, err := godevo.Initialize(minparams, maxparams, 15, parabola)
	if err != nil {
		panic(err)
	}
	model.TrialFunction = godevo.TrialPopulationSP95
	model.CrossoverConstant = 0.4
	model.WeightingFactor = 0.8

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
}

func parabola(vec []float64) float64 {
	res := 3.0 + (vec[0]-1.0)*(vec[0]-1.0) + (vec[1]-2.0)*(vec[1]-2.0)

	return res
}
