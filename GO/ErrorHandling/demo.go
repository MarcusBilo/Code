package main

import (
	"errors"
	"fmt"
	"runtime"
)

func divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, errors.New("division by zero is not allowed")
	}
	return a / b, nil
}

func Log(err error) {
	if err != nil {
		_, _, line, _ := runtime.Caller(1)
		fmt.Println("Error:", err.Error(), "| Line:", line)
	}
}

func main() {
	_, err1 := divide(10, 0)
	Log(err1)

	_, err2 := divide(10, 0)
	if err2 != nil {
		fmt.Println("Error:", err2)
	}

	if _, err3 := divide(10, 0); err3 != nil {
		fmt.Println("Error:", err3)
	}
}
