package models

type Point struct {
	intercept float64
	slope     float64
}

func NewPoint(intercept, slope float64) Point {
	return Point{
		intercept: intercept,
		slope:     slope,
	}
}

func (p *Point) GetIntercept() float64 {
	return p.intercept
}

func (p *Point) GetSlope() float64 {
	return p.slope
}
