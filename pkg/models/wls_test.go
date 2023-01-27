package models

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

const delta = 1.0e-6

func TestWlsModelWithStableWeights(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	wls := NewWlsWithoutWeights(x, y)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), delta)
	assert.InDelta(t, 0.25, point.GetSlope(), delta)
}

func TestWlsModelWithSingleWeight(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	wls := NewWlsWithEqualWeights(x, y, 0.9)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), delta)
	assert.InDelta(t, 0.25, point.GetSlope(), delta)
}

func TestWlsModelWithWeights(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	w := []float64{10, 0, 3, 4, 5, 7, 7}
	wls := NewWlsWithWeights(x, y, w)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 1.1433012, point.GetIntercept(), delta)
	assert.InDelta(t, 0.3962990, point.GetSlope(), delta)
}

func TestRunUphillOk(t *testing.T) {
	x := []float64{0.0, 1.0}
	y := []float64{0.0, 1.0}

	wls := NewWlsWithEqualWeights(x, y, 1)
	point := wls.FitLinearRegression()
	assert.Equal(t, 0.0, point.GetIntercept())
	assert.Equal(t, 1.0, point.GetSlope())
}

func TestPanicIfSequenceHasSingleValue(t *testing.T) {
	x := []float64{1}
	y := []float64{1}

	assert.Panics(t, func() { NewWlsWithoutWeights(x, y) })
}

func TestPanicIfSequencesOfDifferentSize(t *testing.T) {
	x := []float64{1, 2}
	y := []float64{1, 2, 3}

	assert.Panics(t, func() { NewWlsWithoutWeights(x, y) })
}
