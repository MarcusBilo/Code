package main

import (
	"math"
	"runtime/debug"
	"testing"
)

// go test -bench . -benchmem

// go test -bench . -benchmem -memprofile mem.prof
//go tool pprof mem.prof

// go test -bench . -benchmem -cpuprofile cpu.prof
// go tool pprof cpu.prof

func BenchmarkMain(b *testing.B) {
	gcpercent := debug.SetGCPercent(-1)
	memlimit := debug.SetMemoryLimit(math.MaxInt64)
	for i := 0; i < b.N; i++ {
		main()
	}
	debug.SetGCPercent(gcpercent)
	debug.SetMemoryLimit(memlimit)
}
