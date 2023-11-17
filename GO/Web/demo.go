package main

// $ "go get github.com/alexedwards/flow"

import (
	"fmt"
	"github.com/alexedwards/flow"
	"log"
	"net/http"
	"text/template"
)

func main() {
	mux := flow.New()
	mux.Use(htmxMiddleware)
	mux.HandleFunc("/", indexHandler, "GET")
	mux.HandleFunc("/clicked", clickedHandler, "POST")
	err := http.ListenAndServe(":8080", mux)
	if err != nil {
		log.Fatal(err)
	}
}

func htmxMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Header().Set("HX-Current-URL", r.URL.String())
		next.ServeHTTP(w, r)
	})
}

func indexHandler(w http.ResponseWriter, _ *http.Request) {
	renderHTML(w, "index", nil)
}

func clickedHandler(w http.ResponseWriter, _ *http.Request) {
	renderHTML(w, "clicked", nil)
}

func renderHTML(w http.ResponseWriter, tmpl string, data interface{}) {
	tmplPath := fmt.Sprintf("%s.html", tmpl)
	t, err := template.ParseFiles(tmplPath)
	if err != nil {
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	err = t.Execute(w, data)
	if err != nil {
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
}
