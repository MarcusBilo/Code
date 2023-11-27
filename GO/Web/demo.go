package main

import (
	"github.com/alexedwards/flow"
	"log"
	"net/http"
	"text/template"
)

func main() {
	mux := flow.New()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		renderHTML(w, "index.html")
	}, "GET")
	err := http.ListenAndServe(":8080", mux)
	if err != nil {
		log.Fatal(err)
	}
}

func renderHTML(w http.ResponseWriter, fileName string) {
	err := template.Must(template.ParseFiles(fileName)).ExecuteTemplate(w, fileName, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
