package main

import (
	"html/template"
	"log"
	"net/http"
)

func main() {
	indexHandler := func(w http.ResponseWriter, r *http.Request) {
		renderHTML(w, "index.html")
	}
	http.HandleFunc("/", indexHandler)
	err := http.ListenAndServe(":8080", nil)
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
