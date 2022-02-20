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

func TestWlsModelWithWeights(t *testing.T) {

}
