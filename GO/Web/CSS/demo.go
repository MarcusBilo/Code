package main

import (
	"html/template"
	"log"
	"net/http"
)

type PageData struct {
	Language          string
	Title             string
	H1Content         string
	PContent          string
	ButtonContent     string
	AddButton1Content string
	AddButton2Content string
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/htmx_v1.9.10.min.js" {
			w.Header().Set("Content-Type", "application/javascript")
			http.ServeFile(w, r, "htmx_v1.9.10.min.js")
			return
		}
		if r.URL.Path == "/styles.css" {
			w.Header().Set("Content-Type", "text/css")
			http.ServeFile(w, r, "styles.css")
			return
		}
		switch r.URL.Path {
		case "/selawksl.ttf":
			// https://learn.microsoft.com/de-de/typography/font-list/selawik
			http.ServeFile(w, r, "selawksl.ttf")
		case "/favicon.ico":
			// https://www.flaticon.com/free-icon/quality_6294076
			http.ServeFile(w, r, "quality.png")
		case "/3834171_80219_crop.avif":
			// https://www.freepik.com/free-vector/geometric-triangle-pattern-illustration_3834171.htm
			http.ServeFile(w, r, "3834171_80219_crop.avif")
		case "/de.svg":
			// https://flagicons.lipis.dev/
			http.ServeFile(w, r, "de.svg")
		case "/gb.svg":
			// https://flagicons.lipis.dev/
			http.ServeFile(w, r, "gb.svg")
		default:
			http.Redirect(w, r, "/en/index1", http.StatusMovedPermanently)
		}
	})
	http.HandleFunc("/en/index1", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "HTMX & Basic CSS Example",
			H1Content:         "HTML & CSS - Index1EN.html",
			PContent:          "Click the button below to toggle the visibility:",
			ButtonContent:     "Toggle Visibility",
			AddButton1Content: "Switch to 2nd File",
			AddButton2Content: "Switch to 3rd File",
		}
		renderHTML(w, r, "generic_index1.html", data)
	})
	http.HandleFunc("/de/index1", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "HTMX & Basic CSS Beispiel",
			H1Content:         "HTML & CSS - Index1DE.html",
			PContent:          "Auf den Button klicken um die Sichtbarkeit zu ändern:",
			ButtonContent:     "Sichtbarkeit umschalten",
			AddButton1Content: "Wechsel zur 2. Datei",
			AddButton2Content: "Wechsel zur 3. Datei",
		}
		renderHTML(w, r, "generic_index1.html", data)
	})
	http.HandleFunc("/en/index2", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "HTMX & Basic CSS Example",
			H1Content:         "HTML & CSS - Index2EN.html",
			PContent:          "Click the button below to toggle the visibility:",
			ButtonContent:     "Toggle Visibility",
			AddButton1Content: "Switch to 1nd File",
			AddButton2Content: "Switch to 3rd File",
		}
		renderHTML(w, r, "generic_index2.html", data)
	})
	http.HandleFunc("/de/index2", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "HTMX & Basic CSS Beispiel",
			H1Content:         "HTML & CSS - Index2DE.html",
			PContent:          "Auf den Button klicken um die Sichtbarkeit zu ändern:",
			ButtonContent:     "Sichtbarkeit umschalten",
			AddButton1Content: "Wechsel zur 1. Datei",
			AddButton2Content: "Wechsel zur 3. Datei",
		}
		renderHTML(w, r, "generic_index2.html", data)
	})
	http.HandleFunc("/en/index3", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "HTMX & Basic CSS Example",
			AddButton1Content: "Back to 1st File",
			AddButton2Content: "Back to 2nd File",
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	http.HandleFunc("/de/index3", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "HTMX & Basic CSS Beispiel",
			AddButton1Content: "Zurück zur 1. Datei",
			AddButton2Content: "Zurück zur 2. Datei",
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

func renderHTML(w http.ResponseWriter, r *http.Request, templateFile string, data PageData) {
	tmpl, err := template.ParseFiles(templateFile)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	err = tmpl.Execute(w, data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
