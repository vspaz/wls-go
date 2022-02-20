package models

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestWlsModelWithStableWeights(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 2, 3, 4, 5, 6, 7}
	wls := NewWlsWithoutWeights(x, y)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), 6)
	assert.InDelta(t, 0.25, point.GetSlope(), 6)
}

func TestWlsModelWithSingleWeight(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 2, 3, 4, 5, 6, 7}
	wls := NewWlsWithEqualWeights(x, y, 0.9)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), 6)
	assert.InDelta(t, 0.25, point.GetSlope(), 6)
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
