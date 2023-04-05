# wls-go
WLS, weighted linear regression in pure Go w/o any 3d party dependencies or frameworks.

### How-to

```go
package main

import (
	"fmt"
	"github.com/vspaz/wls-go/pkg/models"
)

func main() {
	xPoints := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}  // ∈ {int8, int16, int32, int, float32, float64}
	yPoints := []float64{1.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0}  // ∈ {int8, int16, int32, int, float32, float64}
	weights := []float64{10.0, 1.0, 3.0, 8.0, 14.0, 21.0, 13.0}  // ∈ {int8, int16, int32, int, float32, float64}
	wls := models.NewWlsWithWeights(xPoints, yPoints, weights)  
	point := wls.FitLinearRegression()
	fmt.Println(point.GetSlope())
	fmt.Println(point.GetIntercept())
}
```

## Description

WLS is based on the OLS method and help solve problems of model inadequacy or violations of the basic regression
assumptions.

Estimating a linear regression with WLS is useful, but can appear to be daunting w/o special stats packages, such as
Python statsmodels or Pandas.

## References

- [Wikipedia: Weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares)
- [Introduction to Linear Regression Analysis, 5th edition](https://tinyurl.com/y3clfnrs)
- [Least Squares Regression Analysis in Terms of Linear Algebra](https://tinyurl.com/y485qhlg) 

