package models

type Wls struct {
	y []float64
	x []float64
	w []float64
}

func populateWeights(capacity int, value float64) []float64 {
	weights := make([]float64, capacity, capacity)
	for idx, _ := range weights {
		weights[idx] = value
	}
	return weights
}

func NewWlsWithoutWeights(x []float64, y []float64) Wls {
	if len(x) != len(y) {
		panic("the count of values doesn't match")
	}
	if len(x) < 2 {
		panic("not enough points to fit linear regression")
	}
	return Wls{x: x, y: y, w: populateWeights(len(x), 1.0)}
}

func NewWlsWithWeights(x []float64, y []float64, w []float64) Wls {
	if len(x) != len(y) || len(x) != len(w) {
		panic("the count of values doesn't match")
	}
	return Wls{x: x, y: y, w: w}
}
