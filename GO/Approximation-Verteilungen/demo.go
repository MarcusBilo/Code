package main

import (
	"errors"
	"fmt"
	"math"
	"math/big"
)

func factorial(n int) *big.Int {
	if n == 0 || n == 1 {
		return big.NewInt(1)
	} else {
		return new(big.Int).MulRange(1, int64(n))
	}
}

func binomialCoefficient(n, k int) *big.Int {
	if n == k {
		return big.NewInt(1)
	} else {
		x := factorial(n)
		y := new(big.Int).Mul(factorial(k), factorial(n-k))
		return new(big.Int).Div(x, y)
	}
}

// Returns the sum of the probabilities of the binomial distribution from 1 to k (exclusive)
func cumulativeBinomial(n int, k int, p float64) (*big.Float, error) {
	if n < k || n < 1 || k < 1 || p < 0 || 1 < p {
		return nil, errors.New("invalid input")
	} else {
		var result = big.NewFloat(0)
		for i := 1; i < k; i++ {
			coefficient := new(big.Float).SetInt(binomialCoefficient(n, i))
			success := new(big.Float).SetFloat64(math.Pow(p, float64(i)))
			failure := new(big.Float).SetFloat64(math.Pow(1-p, float64(n-i)))
			probability := new(big.Float).Mul(success, failure)
			subtotal := new(big.Float).Mul(coefficient, probability)
			result = new(big.Float).Add(result, subtotal)
		}
		return result, nil
	}
}

func main() {
	result, err := cumulativeBinomial(60, 15, 0.2)
	if err != nil {
		fmt.Println("Error:", err)
		return
	} else {
		result.SetPrec(15)
		fmt.Println("\nP(X<15) =", result)
		return
	}
}
