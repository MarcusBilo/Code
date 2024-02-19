package main

import (
	"html/template"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/index", http.StatusFound)
	})
	http.HandleFunc("/index", func(w http.ResponseWriter, r *http.Request) {
		renderHTML(w, r, "index.html")
	})
	http.HandleFunc("/index2", func(w http.ResponseWriter, r *http.Request) {
		renderHTML(w, r, "index2.html")
	})
	http.HandleFunc("/index3", func(w http.ResponseWriter, r *http.Request) {
		renderHTML(w, r, "index3.html")
	})
	http.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		// https://www.flaticon.com/free-icon/quality_6294076
		http.ServeFile(w, r, "quality.png")
	})
	http.HandleFunc("/3834171_80219.avif", func(w http.ResponseWriter, r *http.Request) {
		// https://www.freepik.com/free-vector/geometric-triangle-pattern-illustration_3834171.htm
		http.ServeFile(w, r, "3834171_80219.avif")
	})

	http.HandleFunc("/htmx_v1.9.10.min.js", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/javascript")
		http.ServeFile(w, r, "htmx_v1.9.10.min.js")
	})
	http.Handle("/styles.css", http.FileServer(http.Dir("./")))
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

func renderHTML(w http.ResponseWriter, r *http.Request, fileName string) {
	data := map[string]interface{}{
		"RequestPath": r.URL.Path,
	}
	err := template.Must(template.ParseFiles(fileName)).ExecuteTemplate(w, fileName, data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
