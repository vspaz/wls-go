package models

type yTimePoint interface {
	int8 | int16 | int32 | int | int64 | float32 | float64
}

type xTimePoint interface {
	int8 | int16 | int32 | int | int64 | float32 | float64
}

type wTimePoint interface {
	int8 | int16 | int32 | int | int64 | float32 | float64
}

type Wls[X xTimePoint, Y yTimePoint, W wTimePoint] struct {
	x_points []X
	y_points []Y
	weights  []W
}

func populateWeights[W wTimePoint](capacity int, value W) []W {
	weights := make([]W, capacity)
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

func NewWlsWithoutWeights[X xTimePoint, Y yTimePoint](x_points []X, y_points []Y) Wls[X, Y, X] {
	mustHaveSameSize(len(x_points), len(y_points))
	mustHaveSizeGreaterThanTwo(len(x_points))
	return Wls[X, Y, X]{x_points: x_points, y_points: y_points, weights: populateWeights[X](len(x_points), 1.0)}
}

func NewWlsWithWeights[X xTimePoint, Y yTimePoint, W wTimePoint](x_points []X, y_points []Y, weights []W) Wls[X, Y, W] {
	mustHaveSameSize(len(x_points), len(y_points))
	mustHaveSameSize(len(x_points), len(weights))
	mustHaveSizeGreaterThanTwo(len(x_points))
	return Wls[X, Y, W]{x_points: x_points, y_points: y_points, weights: weights}
}

func NewWlsWithStableWeights[X xTimePoint, Y yTimePoint, W wTimePoint](x_points []X, y_points []Y, weights W) Wls[X, Y, W] {
	mustHaveSameSize(len(x_points), len(y_points))
	mustHaveSizeGreaterThanTwo(len(x_points))
	return Wls[X, Y, W]{x_points: x_points, y_points: y_points, weights: populateWeights[W](len(x_points), weights)}
}

func (wls *Wls[X, Y, W]) FitLinearRegression() *Point {
	sumOfWeights := 0.0
	sumOfWeightsByXSquared := 0.0
	sumOfXByYByWeights := 0.0
	sumOfXByWeights := 0.0
	sumOfYByWeights := 0.0

	var xi, yi, wi, xiByWi float64
	for i := 0; i < len(wls.x_points); i++ {
		xi = float64(wls.x_points[i])
		yi = float64(wls.y_points[i])
		wi = float64(wls.weights[i])

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
