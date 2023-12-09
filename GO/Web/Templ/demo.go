package main

import (
	"fmt"
	"github.com/a-h/templ"
	"net/http"
)

// $ "go get github.com/a-h/templ"
// $ "go install github.com/a-h/templ/cmd/templ@latest"
// $ "templ generate"

func main() {
	component := hello("John")
	http.Handle("/", templ.Handler(component))
	fmt.Println("Listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		return
	}
}
