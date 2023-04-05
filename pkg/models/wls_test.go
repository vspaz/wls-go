package models

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

const delta = 1.0e-6

func TestWlsModelWithStableWeightsOk(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	wls := NewWlsWithoutWeights(x, y)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), delta)
	assert.InDelta(t, 0.25, point.GetSlope(), delta)
}

func TestWlsModelWithStableWeightsWidthDifferentDataTypesOk(t *testing.T) {
	x := []int{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	wls := NewWlsWithoutWeights(x, y)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), delta)
	assert.InDelta(t, 0.25, point.GetSlope(), delta)
}

func TestWlsModelWithSingleWeightOk(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	wls := NewWlsWithStableWeights(x, y, 0.9)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), delta)
	assert.InDelta(t, 0.25, point.GetSlope(), delta)
}

func TestWlsModelWithWeightsOk(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	w := []float64{10, 0, 3, 4, 5, 7, 7}
	wls := NewWlsWithWeights(x, y, w)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 1.1433012, point.GetIntercept(), delta)
	assert.InDelta(t, 0.3962990, point.GetSlope(), delta)
}

func TestWlsModelWithWeightsWithDifferentDataTypesOk(t *testing.T) {
	x := []int8{1, 2, 3, 4, 5, 6, 7}
	y := []float32{1.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0}
	w := []float64{10, 0, 3, 4, 5, 7, 7}
	wls := NewWlsWithWeights(x, y, w)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 1.1433012, point.GetIntercept(), delta)
	assert.InDelta(t, 0.3962990, point.GetSlope(), delta)
}

func TestHorizontalLineOk(t *testing.T) {
	x := []float64{0.0, 1.0}
	y := []float64{10.0, 10.0}

	wls := NewWlsWithStableWeights(x, y, 1)
	point := wls.FitLinearRegression()
	assert.Equal(t, 10.0, point.GetIntercept())
	assert.Equal(t, 0.0, point.GetSlope())
}

func TestRunUphillOk(t *testing.T) {
	x := []float64{0.0, 1.0}
	y := []float64{0.0, 1.0}

	wls := NewWlsWithStableWeights(x, y, 1)
	point := wls.FitLinearRegression()
	assert.Equal(t, 0.0, point.GetIntercept())
	assert.Equal(t, 1.0, point.GetSlope())
}

func TestRunDownhillOk(t *testing.T) {
	x := []float64{1.0, 0.0}
	y := []float64{0.0, 1.0}

	wls := NewWlsWithStableWeights(x, y, 1)
	point := wls.FitLinearRegression()
	assert.Equal(t, 1.0, point.GetIntercept())
	assert.Equal(t, -1.0, point.GetSlope())
}

func TestSequenceHasSingleValueFail(t *testing.T) {
	x := []float64{1}
	y := []float64{1}

	assert.Panics(t, func() { NewWlsWithoutWeights(x, y) })
}

func TestSequencesOfDifferentSizeFail(t *testing.T) {
	x := []float64{1, 2}
	y := []float64{1, 2, 3}

	assert.Panics(t, func() { NewWlsWithoutWeights(x, y) })
}
