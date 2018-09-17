// Differential Evolution Genetic Algorythm
// This routine minimises a vector of parameters using differential
// evolution as described by Kenneth Price and Rainer Storn in Dr. Dobb's
// Journal #264, 1997. pps 18-24,78.
// Also in "Differential Evolution - A simple and efficient adaptive
// scheme for global optimization over continuous spaces" vt Storn & Price
// 1995 and "Differential Evolution – A Simple and Efficient Heuristic for
// Global Optimization over Continuous Spaces" Storn & Price, Journal of
// Global Optimization 11:341-359, 1997.
// Implemented in Go by Christopher De Vries (C) 2018.

// Package godevo is a differential evolution and Markov Chain Monte Carlo
// differential evolution library.
package godevo

import (
	"errors"
	"math"
	"math/rand"
)

// Model is a differential evolution model structure
type Model struct {
	Population        [][]float64                                     // The population of parameters
	Fitness           []float64                                       // The fitness of each parameter set
	CrossoverConstant float64                                         // The crossover constant: CR
	WeightingFactor   float64                                         // The weighting factor: F
	DenialFunction    func(float64, float64) bool                     // The denial function, either GreedyDenial or MetropolisDenial
	TrialFunction     func([][]float64, float64, float64) [][]float64 // The trial population function, TrialPopulationSP97 for Storn & Price (1997)
	ModelFunction     func([]float64) float64                         // The function to optimize
	ParallelMode      bool                                            // Do fitness calculation in parallel
}

// TrialPopulationSP97 will generate a trial population according to Storn & Price 1997 paper.
// The population is the previous generation population of parameters.
// The f parameter is the weighting factor in the differential evolution algorithm.
// The cr parameter is the crossover constance in the differential evolution algorithm.
// The function returns a new trial population of parameters.
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

// TrialPopulationSP95 will generate a trial population according to Storn & Price 1995 notes.
// The population is the previous generation population of parameters.
// The f parameter is the weighting factor in the differential evolution algorithm.
// The cr parameter is the crossover constance in the differential evolution algorithm.
// The function returns a new trial population of parameters.
func TrialPopulationSP95(population [][]float64, f float64, cr float64) [][]float64 {
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

		l := 1
		for rand.Float64() < cr && l < d {
			l++
		}

		copy(nextpopulation[i], population[i])
		for k := s; k < s+l; k++ {
			nextpopulation[i][k%d] = population[a][k%d] + f*(population[b][k%d]-population[c][k%d])
		}
	}

	return nextpopulation
}

// TrialPopulationParent will enerate a trial population according to Storn & Price 1997 paper,
// but always evolve from parent vector at same position.
// The population is the previous generation population of parameters.
// The f parameter is the weighting factor in the differential evolution algorithm.
// The cr parameter is the crossover constance in the differential evolution algorithm.
// The function returns a new trial population of parameters.
func TrialPopulationParent(population [][]float64, f float64, cr float64) [][]float64 {
	np := len(population)
	d := len(population[0])

	nextpopulation := make([][]float64, np)

	for i := 0; i < np; i++ {
		nextpopulation[i] = make([]float64, d)
		a := i
		b := i
		c := i

		for b == a {
			b = rand.Intn(np)
		}
		for c == a || c == b {
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

// GreedyDenial is the standard differential evolution denial function which returns True if the new fitness
// value is greater than the old fitness value.
func GreedyDenial(oldFitness float64, newFitness float64) bool {
	return newFitness >= oldFitness
}

// MetropolisDenial is a function which has a greater probability of returning True if the
// new fitness value is large compared to the old fitness value.
func MetropolisDenial(oldFitness float64, newFitness float64) bool {
	dt := math.Exp((oldFitness - newFitness) / 2.0)

	result := dt <= rand.Float64()
	return result
}

// Initialize returns a standard initialized differential evolution model with a population of np parameters.
// pmin are the minimum parameter values, and pmax are the maximum parameter values. The parallel parameter
// controls if the fitness function is computed in parallel or not. The function to
// be optimized is modelFunction.
func Initialize(pmin []float64, pmax []float64, np int, parallel bool, modelFunction func([]float64) float64) (*Model, error) {
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

	fitness := make([]float64, np)

	if parallel {
		parallelCalculateFitness(&result, &fitness, modelFunction)
	} else {
		calculateFitness(&result, &fitness, modelFunction)
	}

	model := Model{
		Population:        result,
		Fitness:           fitness,
		CrossoverConstant: 0.1,
		WeightingFactor:   0.7,
		DenialFunction:    GreedyDenial,
		TrialFunction:     TrialPopulationSP97,
		ModelFunction:     modelFunction,
		ParallelMode:      parallel,
	}

	return &model, nil
}

// InitializeMCMC returns an initialized Markov chain Mote Carlo differential evolution model with a population of np parameters.
// pmin are the minimum parameter values, and pmax are the maximum parameter values. The parallel parameter controls
// if the optimization function is to be run in parallel. The function to
// be optimized is modelFunction.
func InitializeMCMC(pmin []float64, pmax []float64, np int, parallel bool, modelFunction func([]float64) float64) (*Model, error) {
	model, err := Initialize(pmin, pmax, np, parallel, modelFunction)
	if err != nil {
		return nil, err
	}

	model.DenialFunction = MetropolisDenial
	model.TrialFunction = TrialPopulationParent

	return model, nil
}

// Step forward one generation in differential evolution modeling.
func (model *Model) Step() {
	trialPopulation := model.TrialFunction(model.Population, model.WeightingFactor, model.CrossoverConstant)
	trialFitness := make([]float64, len(trialPopulation))

	if model.ParallelMode {
		parallelCalculateFitness(&trialPopulation, &trialFitness, model.ModelFunction)
	} else {
		calculateFitness(&trialPopulation, &trialFitness, model.ModelFunction)
	}

	for i := range trialFitness {
		denial := model.DenialFunction(model.Fitness[i], trialFitness[i])
		if !denial {
			model.Fitness[i] = trialFitness[i]
			model.Population[i] = trialPopulation[i]
		}
	}
}

// Best returns the optimal parameters, and the fitness of the optimal parameters from the model.
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

// MeanStd returns the mean and standard deviation of the parameters of the population
func (model *Model) MeanStd() ([]float64, []float64) {
	meanParameters := make([]float64, len(model.Population[0]))
	standardDeviation := make([]float64, len(model.Population[0]))

	np := len(model.Population)

	for _, v := range model.Population {
		for i := range v {
			meanParameters[i] += v[i]
			standardDeviation[i] += v[i] * v[i]
		}
	}

	for i := range meanParameters {
		meanParameters[i] /= float64(np)
		standardDeviation[i] = math.Sqrt(standardDeviation[i]/float64(np) - meanParameters[i]*meanParameters[i])
	}

	return meanParameters, standardDeviation
}

// calculateFitness calculates the fiteness of a particular parameter set and place it in the fitness array. Both come from
// the location in the slices.
func calculateFitness(population *[][]float64, fitness *[]float64, modelFunction func([]float64) float64) {
	for i := range *fitness {
		parameters := (*population)[i]

		fitnessValue := modelFunction(parameters)
		(*fitness)[i] = fitnessValue
	}
}

type fitnessResult struct {
	Index  int
	Result float64
}

func parallelCalculateFitness(population *[][]float64, fitness *[]float64, modelFunction func([]float64) float64) {
	ch := make(chan fitnessResult)

	for i := range *fitness {
		go func(i int, parameters []float64, rchan chan<- fitnessResult) {
			var result fitnessResult
			result.Index = i
			result.Result = modelFunction(parameters)
			rchan <- result
		}(i, (*population)[i], ch)
	}

	for range *fitness {
		r := <-ch
		(*fitness)[r.Index] = r.Result
	}
}
