package main

import (
	"fmt"
	"github.com/vspaz/wls-go/pkg/models"
)

func main() {
	xPoints := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	yPoints := []float64{1.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0}
	weights := []float64{10.0, 1.0, 3.0, 8.0, 14.0, 21.0, 13.0}
	wls := models.NewWlsWithWeights(xPoints, yPoints, weights)
	point := wls.FitLinearRegression()
	fmt.Println(point.GetSlope())
	fmt.Println(point.GetIntercept())
}
