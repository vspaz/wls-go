package main

import (
	"fmt"
	"github.com/vspaz/wls-go/pkg/models"
)

func main() {
	x := []float64{1, 2, 3, 4, 5, 6, 7}
	y := []float64{1, 2, 3, 4, 5, 6, 7}
	w := []float64{1, 2, 3, 4, 5, 6, 7}
	wls := models.NewWlsWithWeights(x, y, w)
	point := wls.FitLinearRegression()
	fmt.Println(point.GetSlope())
	fmt.Println(point.GetIntercept())
}
