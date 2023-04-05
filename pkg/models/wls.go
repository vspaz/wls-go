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
	xPoints []X
	yPoints []Y
	weights []W
}

func populateWeights[T wTimePoint](capacity int, value T) []T {
	weights := make([]T, capacity)
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

func NewWlsWithoutWeights[X xTimePoint, Y yTimePoint](xPoints []X, yPoints []Y) Wls[X, Y, X] {
	mustHaveSameSize(len(xPoints), len(yPoints))
	mustHaveSizeGreaterThanTwo(len(xPoints))
	return Wls[X, Y, X]{xPoints: xPoints, yPoints: yPoints, weights: populateWeights[X](len(xPoints), 1.0)}
}

func NewWlsWithWeights[X xTimePoint, Y yTimePoint, W wTimePoint](xPoints []X, yPoints []Y, weights []W) Wls[X, Y, W] {
	mustHaveSameSize(len(xPoints), len(yPoints))
	mustHaveSameSize(len(xPoints), len(weights))
	mustHaveSizeGreaterThanTwo(len(xPoints))
	return Wls[X, Y, W]{xPoints: xPoints, yPoints: yPoints, weights: weights}
}

func NewWlsWithStableWeights[X xTimePoint, Y yTimePoint, W wTimePoint](xPoints []X, yPoints []Y, weights W) Wls[X, Y, W] {
	mustHaveSameSize(len(xPoints), len(yPoints))
	mustHaveSizeGreaterThanTwo(len(xPoints))
	return Wls[X, Y, W]{xPoints: xPoints, yPoints: yPoints, weights: populateWeights[W](len(xPoints), weights)}
}

func (wls *Wls[X, Y, W]) FitLinearRegression() *Point {
	sumOfWeights := 0.0
	sumOfWeightsByXSquared := 0.0
	sumOfXByYByWeights := 0.0
	sumOfXByWeights := 0.0
	sumOfYByWeights := 0.0

	var xi, yi, wi, xiByWi float64
	for i := 0; i < len(wls.xPoints); i++ {
		xi = float64(wls.xPoints[i])
		yi = float64(wls.yPoints[i])
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

	return NewPoint(intercept, slope)
}
