package main

import (
	"fmt"
	"math/rand"
	"sync"
)

func main() {

	numbers := rand.Perm(101) // 0,1,...,99,100
	numWorkers := 2
	ch := make(chan int)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			chunkSize := len(numbers) / numWorkers
			start := workerID * chunkSize
			end := (workerID + 1) * chunkSize
			if workerID == numWorkers-1 {
				end = len(numbers)
			}
			sum := 0
			for j := start; j < end; j++ {
				sum += numbers[j]
			}
			ch <- sum
		}(i)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	totalSum := 0
	for partialSum := range ch {
		totalSum += partialSum
	}

	fmt.Println("\nTotal Sum:", totalSum)
}
