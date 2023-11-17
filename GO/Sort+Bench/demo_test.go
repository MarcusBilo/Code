package main

import (
	"math/rand"
	"slices"
	"testing"
)

// $ "go test -bench . -benchmem"

var rng = rand.New(rand.NewSource(42)) // fixed seed for consistency

func generateRandomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := 0; i < size; i++ {
		// Generate a float64 between 0 and 1000 with 2 decimal places
		arr[i] = float64(rng.Intn(1e5)) / 100.0
	}
	return arr
}

func BenchmarkFunction1e5merge(b *testing.B) {
	arr := generateRandomArray(1e5)
	for i := 0; i < b.N; i++ {
		copyArr := make([]float64, len(arr))
		copy(copyArr, arr)
		mergeSort(copyArr)
	}
}

func BenchmarkFunction1e5built(b *testing.B) {
	arr := generateRandomArray(1e5)
	for i := 0; i < b.N; i++ {
		copyArr := make([]float64, len(arr))
		copy(copyArr, arr)
		slices.Sort(arr)
	}
}
