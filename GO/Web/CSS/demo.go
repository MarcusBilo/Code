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
	http.HandleFunc("/favicon.ico", noFavicon)
	http.HandleFunc("/htmx_v1.9.10.min.js", htmxHandler)
	http.Handle("/styles.css", http.FileServer(http.Dir("./")))
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

func noFavicon(w http.ResponseWriter, r *http.Request) {
	// This handler does nothing, effectively returning no favicon
}

func htmxHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/javascript")
	http.ServeFile(w, r, "htmx_v1.9.10.min.js") // Replace with the actual path
}

func renderHTML(w http.ResponseWriter, fileName string) {
	err := template.Must(template.ParseFiles(fileName)).ExecuteTemplate(w, fileName, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
