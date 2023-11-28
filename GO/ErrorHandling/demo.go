package main

import (
	"errors"
	"fmt"
	"runtime"
	"time"
)

func divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, errors.New("division by zero is not allowed")
	} else {
		return a / b, nil
	}
}

func Log(err error) {
	if err != nil {
		_, _, line, _ := runtime.Caller(1)
		timestamp := time.Now().Format("2006-01-02 15:04:05")
		fmt.Println(timestamp, "- Error:", err.Error(), "| Line:", line)
		return
	} else {
		return
	}
}

func main() {
	var err error

	_, err = divide(10, 0)
	Log(err)

	_, err = divide(10, 0)
	if err != nil {
		fmt.Println("Error:", err)
	}

	if _, err = divide(10, 0); err != nil {
		fmt.Println("Error:", err)
	}

	return
}
