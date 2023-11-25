package main

import (
	"fmt"
	"math"
	"math/big"
)

func factorial(n int) *big.Int {
	result := new(big.Int).MulRange(1, int64(n))
	return result
}

func binomialCoefficient(n, k int) *big.Int {
	numerator := factorial(n)
	denominator := new(big.Int).Mul(factorial(k), factorial(n-k))
	result := new(big.Int).Div(numerator, denominator)
	return result
}

func cumulativeBinomial(n int, k int, p float64) *big.Float {
	var result = new(big.Float).SetFloat64(0.0)
	for i := 1; i < k; i++ {
		coefficient := new(big.Float).SetInt(binomialCoefficient(n, i))
		success := new(big.Float).SetFloat64(math.Pow(p, float64(i)))
		failure := new(big.Float).SetFloat64(math.Pow(1-p, float64(n-i)))
		probability := new(big.Float).Mul(success, failure)
		subtotal := new(big.Float).Mul(coefficient, probability)
		result = new(big.Float).Add(result, subtotal)
	}
	return result
}

func main() {
	result := cumulativeBinomial(60, 15, 0.2).SetPrec(15)
	fmt.Println("\nP(X<15) =", result, "%")
}
