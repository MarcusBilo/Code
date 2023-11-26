package main

import (
	"errors"
	"fmt"
	"math"
	"math/big"
)

func factorial(n uint) *big.Int {
	if n == 0 || n == 1 {
		return big.NewInt(1)
	} else {
		return new(big.Int).MulRange(1, int64(n))
	}
}

func binomialCoefficient(n uint, k uint) *big.Int {
	if n == k {
		return big.NewInt(1)
	} else {
		x := factorial(n)
		y := new(big.Int).Mul(factorial(k), factorial(n-k))
		return new(big.Int).Div(x, y)
	}
}

// Returns the sum of the probabilities of the binomial distribution from 0 to k (inclusive).
func cumulativeBinomial(n uint, k uint, p float64) (*big.Float, error) {
	if n < k || n == 0 || k == 0 || p < 0 || 1 < p {
		return nil, errors.New("invalid input")
	} else {
		var result = new(big.Float).SetPrec(64).SetFloat64(0)
		var i uint
		for i = 0; i <= k; i++ {
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

// Returns the probability that a variable assumes a value less than or equal to k. A continuity
// correction of +0.5 is applied. Phi calculation is interchangeable with a Z-Table lookup.
func standardNormalDistribution(n uint, k uint, p float64) (float64, error) {
	if n < k || n == 0 || p < 0 || 1 < p {
		return 0, errors.New("invalid input")
	} else {
		expectedValue := float64(n) * p
		variance := float64(n) * p * (1 - p)
		x := float64(k) + 0.5
		z := (x - expectedValue) / math.Sqrt(variance)
		phi := 0.5 * (1 + math.Erf(z/math.Sqrt2))
		return phi, nil
	}
}

func round(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

func main() {
	if resultBinomial, err := cumulativeBinomial(60, 14, 0.2); err != nil {
		fmt.Println("Error:", err)
	} else {
		resultBinomial.SetPrec(15)
		fmt.Println("\nBinomial distribution: P(X<=14) =", resultBinomial)
	}
	if resultNormal, err := standardNormalDistribution(60, 14, 0.2); err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Println("Normal Approximation:  P(X<=14) =", round(resultNormal, 5))
	}
}
