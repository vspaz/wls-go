package main

import (
	"fmt"
	"github.com/vspaz/wls-go/pkg/models"
)

func main() {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	y := []float64{1.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0}
	w := []float64{10.0, 1.0, 3.0, 8.0, 14.0, 21.0, 13.0}
	wls := models.NewWlsWithWeights(x, y, w)
	point := wls.FitLinearRegression()
	fmt.Println(point.GetSlope())
	fmt.Println(point.GetIntercept())
}
