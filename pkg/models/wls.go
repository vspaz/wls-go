package models

type Wls struct {
	y []float64
	x []float64
	w []float64
}

func populateWeights(capacity int, value float64) []float64 {
	weights := make([]float64, capacity)
	for idx := range weights {
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

func NewWlsWithEqualWeights(x []float64, y []float64, w float64) Wls {
	if len(x) != len(y) {
		panic("the count of values doesn't match")
	}
	return Wls{x: x, y: y, w: populateWeights(len(x), w)}
}

func (wls *Wls) FitLinearRegression() *Point {
	sumOfWeights := 0.0
	sumOfWeightsByXSquared := 0.0
	sumOfXByYByWeights := 0.0
	sumOfXByWeights := 0.0
	sumOfYByWeights := 0.0

	var xi, yi, wi, xiByWi float64
	for i := 0; i < len(wls.x); i++ {
		xi = wls.x[i]
		yi = wls.y[i]
		wi = wls.w[i]

		sumOfWeights += wi
		xiByWi = xi * wi
		sumOfXByWeights += xiByWi
		sumOfXByYByWeights += xiByWi * yi
		sumOfYByWeights += yi * wi
		sumOfWeightsByXSquared += xiByWi * xi
	}

	dividend := sumOfWeights*sumOfXByYByWeights - sumOfXByWeights*sumOfYByWeights
	divisor := sumOfWeights*sumOfWeightsByXSquared - sumOfXByWeights*sumOfXByWeights

	if divisor == 0 {
		return nil
	}

	slope := dividend / divisor
	intercept := (sumOfYByWeights - slope*sumOfXByWeights) / sumOfWeights

	return &Point{
		intercept: intercept,
		slope:     slope,
	}
}
