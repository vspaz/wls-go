package models

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestWlsModelWithStableWeights(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 3, 4, 5, 2, 3, 4}
	wls := NewWlsWithoutWeights(x, y)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), 0.000001)
	assert.InDelta(t, 0.25, point.GetSlope(), 0.000001)
}

func TestWlsModelWithSingleWeight(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 2, 3, 4, 5, 6, 7}
	wls := NewWlsWithEqualWeights(x, y, 0.9)
	point := wls.FitLinearRegression()
	assert.InDelta(t, 2.14285714, point.GetIntercept(), 6)
	assert.InDelta(t, 0.25, point.GetSlope(), 6)
}

func TestWlsModelWithWeights(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 2, 3, 4, 5, 6, 7}
	w := []float64{10, 0, 3, 4, 5, 7, 7}
	wls := NewWlsWithWeights(x, y, w)
	point := wls.FitLinearRegression()
	fmt.Println(point.slope)
	fmt.Println(point.intercept)
	assert.InDelta(t, 0, point.GetIntercept(), 0)
	assert.InDelta(t, 1, point.GetSlope(), 0)
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
