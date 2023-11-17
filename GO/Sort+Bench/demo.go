package main

func mergeSort(arr []float64) {
	if len(arr) > 1 {
		mid := len(arr) / 2
		leftHalf := arr[:mid]
		rightHalf := arr[mid:]
		mergeSort(leftHalf)
		mergeSort(rightHalf)
		i, j, k := 0, 0, 0
		for i < len(leftHalf) && j < len(rightHalf) {
			if leftHalf[i] < rightHalf[j] {
				arr[k] = leftHalf[i]
				i++
			} else {
				arr[k] = rightHalf[j]
				j++
			}
			k++
		}
		for i < len(leftHalf) {
			arr[k] = leftHalf[i]
			i++
			k++
		}
		for j < len(rightHalf) {
			arr[k] = rightHalf[j]
			j++
			k++
		}
	}
}

func main() {
}
