package models

type Wls struct {
	y_points []float64
	x_points []float64
	weights  []float64
}

func populateWeights(capacity int, value float64) []float64 {
	weights := make([]float64, capacity)
	for idx := range weights {
		weights[idx] = value
	}
	return weights
}

func mustHaveSameSize(sizeOne, sizeTwo int) {
	if sizeOne != sizeTwo {
		panic("the count of values doesn't match")
	}
}

func mustHaveSizeGreaterThanTwo(sequenceSize int) {
	if sequenceSize < 2 {
		panic("not enough points to fit linear regression")
	}
}

func NewWlsWithoutWeights(x_points []float64, y_points []float64) Wls {
	mustHaveSameSize(len(x_points), len(y_points))
	mustHaveSizeGreaterThanTwo(len(x_points))
	return Wls{x_points: x_points, y_points: y_points, weights: populateWeights(len(x_points), 1.0)}
}

func NewWlsWithWeights(x_points []float64, y_points []float64, weights []float64) Wls {
	mustHaveSameSize(len(x_points), len(y_points))
	mustHaveSameSize(len(x_points), len(weights))
	mustHaveSizeGreaterThanTwo(len(x_points))
	return Wls{x_points: x_points, y_points: y_points, weights: weights}
}

func NewWlsWithEqualWeights(x_points []float64, y_points []float64, weights float64) Wls {
	mustHaveSameSize(len(x_points), len(y_points))
	mustHaveSizeGreaterThanTwo(len(x_points))
	return Wls{x_points: x_points, y_points: y_points, weights: populateWeights(len(x_points), weights)}
}

func (wls *Wls) FitLinearRegression() *Point {
	sumOfWeights := 0.0
	sumOfWeightsByXSquared := 0.0
	sumOfXByYByWeights := 0.0
	sumOfXByWeights := 0.0
	sumOfYByWeights := 0.0

	var xi, yi, wi, xiByWi float64
	for i := 0; i < len(wls.x_points); i++ {
		xi = wls.x_points[i]
		yi = wls.y_points[i]
		wi = wls.weights[i]

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
