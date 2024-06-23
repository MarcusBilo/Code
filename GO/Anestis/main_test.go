package main

import (
	"math"
	"runtime/debug"
	"testing"
)

// go test -bench . -benchmem

func BenchmarkMain(b *testing.B) {
	for i := 0; i < b.N; i++ {
		main()
	}
}
