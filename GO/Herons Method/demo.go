package main

import (
	"fmt"
	"math"
)

func squareRoot(a float64, decimalPlaces int) float64 {
	epsilon := 1 / math.Pow10(decimalPlaces)
	x := a
	for {
		b := a / x
		x = (x + b) / 2
		if math.Abs(x-b) < epsilon {
			break
		}
	}
	return x
}

func main() {
	a := 1.2          // input: radicand
	minAccuracy := 15 // input: in decimal places
	x := squareRoot(a, minAccuracy)
	fmt.Println(x)
}
