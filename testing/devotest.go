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
	model.CrossoverConstant = 0.4
	model.WeightingFactor = 0.8

	for i := 0; i < 200; i++ {
		model.Step()
		fmt.Println(model.Fitness)
	}

	bp, bf := model.Best()
	fmt.Printf("Best Parameters: %v\n", bp)
	fmt.Printf("Best Fitness: %f\n", bf)
}

func parabola(vec []float64) float64 {
	res := 3.0 + (vec[0]-1.0)*(vec[0]-1.0) + (vec[1]-2.0)*(vec[1]-2.0)

	return res
}
