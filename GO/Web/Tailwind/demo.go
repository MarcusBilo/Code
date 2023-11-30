package main

import (
	"encoding/json"
	"html/template"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", index)
	http.HandleFunc("/data", data)
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

func index(w http.ResponseWriter, _ *http.Request) {
	renderHTML(w, "index.html")
}

func data(w http.ResponseWriter, _ *http.Request) {
	content := map[string]string{"Message": "This is data loaded via HTMX from Go!"}
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(content)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func renderHTML(w http.ResponseWriter, fileName string) {
	err := template.Must(template.ParseFiles(fileName)).ExecuteTemplate(w, fileName, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
