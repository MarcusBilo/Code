package main

import (
	"fmt"
	"github.com/shopspring/decimal"
)

func main() {

	var (
		fullPart int64 = 123456789123456789
		centPart int64 = 123456789123456789
	)
	fmt.Println()
	fullPartDecimal := decimal.NewFromInt(fullPart)
	centPartDecimal := decimal.NewFromInt(centPart)
	decimal.DivisionPrecision = lenLoop(centPart)
	centPartDecimal = centPartDecimal.Div(Pow10(lenLoop(centPart)))
	fmt.Println(fullPartDecimal.Add(centPartDecimal)) // correct 36-digit decimal number

}

func lenLoop(i int64) int {
	if i == 0 {
		return 1
	}
	var count int = 0
	for i != 0 {
		i /= 10
		count++
	}
	return count
}

func Pow10(n int) decimal.Decimal {
	// If n is 0, the result is 1, since 10^0 = 1
	if n == 0 {
		return decimal.NewFromInt(1)
	}

	// If n is positive, calculate 10^n
	if n > 0 {
		result := decimal.NewFromInt(1)
		ten := decimal.NewFromInt(10)
		for i := 0; i < n; i++ {
			result = result.Mul(ten)
		}
		return result
	}

	// If n is negative, calculate 10^n which is 1 / (10^-n)
	result := decimal.NewFromInt(1)
	ten := decimal.NewFromInt(10)
	for i := 0; i < -n; i++ {
		result = result.Div(ten)
	}
	return result
}
