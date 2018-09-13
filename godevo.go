package godevo

import (
	"errors"
	"math"
	"math/rand"
	"sync"
)

type Model struct {
	Population        [][]float64
	Fitness           []float64
	CrossoverConstant float64
	WeightingFactor   float64
	DenialFunction    func(float64, float64) bool
	TrialFunction     func([][]float64, float64, float64) [][]float64
	ModelFunction     func([]float64) float64
}

// Generate a trial population according to Storn & Price 1997 paper.
//
// population - ([][]float64) a slice of slice of parameters.
// f - (float64) The weighting factor in the differential evolution algorithm.
// cr - (float64) The crossover constance in the differential evolution algorithm.
//
// Returns: ([][]float64) A new population
func TrialPopulationSP97(population [][]float64, f float64, cr float64) [][]float64 {
	np := len(population)
	d := len(population[0])

	nextpopulation := make([][]float64, np)

	for i := 0; i < np; i++ {
		nextpopulation[i] = make([]float64, d)
		a := i
		b := i
		c := i

		for a == i {
			a = rand.Intn(np)
		}
		for b == i || b == a {
			b = rand.Intn(np)
		}
		for c == i || c == a || c == b {
			c = rand.Intn(np)
		}

		s := rand.Intn(d)

		copy(nextpopulation[i], population[i])
		for k := 0; k < d; k++ {
			if k == s || rand.Float64() < cr {
				nextpopulation[i][k] = population[a][k] + f*(population[b][k]-population[c][k])
			}
		}

	}

	return nextpopulation
}

// Standard differential evolution denial function which picks the lowest fitness value.
func GreedyDenial(oldFitness float64, newFitness float64) bool {
	return newFitness >= oldFitness
}

// Metropolis denial function which allows higher value if necessary.
func MetropolisDenial(oldFitness float64, newFitness float64) bool {
	dt := math.Exp((oldFitness - newFitness) / 2.0)

	result := dt <= rand.Float64()
	return result
}

func Initialize(pmin []float64, pmax []float64, np int, modelFunction func([]float64) float64) (*Model, error) {
	if len(pmin) != len(pmax) {
		return nil, errors.New("Initial population limit sizes don't match")
	}

	result := make([][]float64, np)
	for i := 0; i < np; i++ {
		result[i] = make([]float64, len(pmin))

		for j := 0; j < len(pmin); j++ {
			result[i][j] = (pmax[j]-pmin[j])*rand.Float64() + pmin[j]
		}
	}

	var wg sync.WaitGroup // Wait group for initializing fitnesses
	fitness := make([]float64, np)

	for i := range result {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			CalculateFitness(&result, &fitness, i, modelFunction)
		}(i)
	}

	wg.Wait()

	model := Model{
		Population:        result,
		Fitness:           fitness,
		CrossoverConstant: 0.1,
		WeightingFactor:   0.7,
		DenialFunction:    GreedyDenial,
		TrialFunction:     TrialPopulationSP97,
		ModelFunction:     modelFunction,
	}

	return &model, nil
}

func (model *Model) Step() {
	trialPopulation := model.TrialFunction(model.Population, model.WeightingFactor, model.CrossoverConstant)
	var wg sync.WaitGroup
	trialFitness := make([]float64, len(trialPopulation))

	for i := range trialPopulation {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			CalculateFitness(&trialPopulation, &trialFitness, i, model.ModelFunction)
		}(i)
	}

	wg.Wait()

	for i := range trialFitness {
		denial := model.DenialFunction(model.Fitness[i], trialFitness[i])
		if !denial {
			model.Fitness[i] = trialFitness[i]
			model.Population[i] = trialPopulation[i]
		}
	}
}

func (model *Model) Best() ([]float64, float64) {
	bestParams := model.Population[0]
	bestFitness := model.Fitness[0]

	for i := 1; i < len(model.Fitness); i++ {
		if bestFitness > model.Fitness[i] {
			bestFitness = model.Fitness[i]
			bestParams = model.Population[i]
		}
	}

	return bestParams, bestFitness
}

func CalculateFitness(population *[][]float64, fitness *[]float64, location int, modelFunction func([]float64) float64) {
	parameters := (*population)[location]

	fitnessValue := modelFunction(parameters)
	(*fitness)[location] = fitnessValue
}
