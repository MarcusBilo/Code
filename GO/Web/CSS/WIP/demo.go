package main

import (
	"fmt"
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

type CardData struct {
	Year        int
	Month       int
	Title       string
	Description string
	Get         string
	Blog        string
}

var enCardDataMap = map[int]CardData{
	1: {
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Title",
		Description: "Description for Card 1.",
		Get:         "./card/1",
		Blog:        "Read Blog",
	},
	2: {
		Year:        2022,
		Month:       2,
		Title:       "Ipsum Card 2 Title",
		Description: "Description for Card 2.",
		Get:         "./card/2",
		Blog:        "Read Blog",
	},
}

var deCardDataMap = map[int]CardData{
	1: {
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Titel",
		Description: "Beschreibung für Card 1.",
		Get:         "./card/1",
		Blog:        "Blog lesen",
	},
	2: {
		Year:        2022,
		Month:       2,
		Title:       "Ipsum Card 2 Titel",
		Description: "Beschreibung für Card 2.",
		Get:         "./card/2",
		Blog:        "Blog lesen",
	},
}

var enBlogDataMap = map[int]PageData{
	1: {
		Language:          "en",
		Title:             "HTMX & Basic CSS Example",
		AddButton1Content: "Back to 1st File",
		AddButton2Content: "Back to 2nd File",
	},
	2: {
		Language:          "en",
		Title:             "HTMX & Basic CSS 2 Example",
		AddButton1Content: "Back to 1st File",
		AddButton2Content: "Back to 2nd File",
	},
}

var deBlogDataMap = map[int]PageData{
	1: {
		Language:          "de",
		Title:             "HTMX & Basic CSS Beispiel",
		AddButton1Content: "Zurück zur 1. Datei",
		AddButton2Content: "Zurück zur 2. Datei",
	},
	2: {
		Language:          "de",
		Title:             "HTMX & Basic CSS 2 Beispiel",
		AddButton1Content: "Zurück zur 1. Datei",
		AddButton2Content: "Zurück zur 2. Datei",
	},
}

func main() {

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/styles.css":
			w.Header().Set("Content-Type", "text/css")
			http.ServeFile(w, r, "styles.css")
		case "/htmx_v1.9.10.min.js":
			// https://unpkg.com/browse/htmx.org@1.9.10/dist/
			w.Header().Set("Content-Type", "application/javascript")
			http.ServeFile(w, r, "htmx_v1.9.10.min.js")
		case "/noun-home-5487412.svg":
			// https://thenounproject.com/icon/home-5487412/
			http.ServeFile(w, r, "noun-home-5487412.svg")
		case "/Noto-Sans-regular.woff2":
			// https://github.com/pages-themes/minimal/blob/master/assets/fonts/Noto-Sans-regular/Noto-Sans-regular.woff2
			http.ServeFile(w, r, "Noto-Sans-regular.woff2")
		case "/JetBrainsMono-Regular.woff2":
			// https://github.com/JetBrains/JetBrainsMono/blob/master/fonts/webfonts/JetBrainsMono-Regular.woff2
			http.ServeFile(w, r, "JetBrainsMono-Regular.woff2")
		case "/3834171_80219_crop.avif":
			// https://www.freepik.com/free-vector/geometric-triangle-pattern-illustration_3834171.htm
			http.ServeFile(w, r, "3834171_80219_crop.avif")
		case "/de.svg":
			// https://flagicons.lipis.dev/
			http.ServeFile(w, r, "de.svg")
		case "/gb.svg":
			// https://flagicons.lipis.dev/
			http.ServeFile(w, r, "gb.svg")
		case "/max-card-number":
			maxNumber := len(enCardDataMap)
			w.Header().Set("Content-Type", "text/plain")
			_, err := fmt.Fprintf(w, "%d", maxNumber)
			if err != nil {
				http.Error(w, "Error writing response", http.StatusInternalServerError)
				return
			}
		default:
			http.Redirect(w, r, "/en/index1", http.StatusMovedPermanently)
		}
	})

	// ########################
	// ########################
	// ########################

	http.HandleFunc("/en/cards/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/en/cards/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		data, ok := enCardDataMap[cardNumber]
		if !ok {
			http.Error(w, "Card not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, "card-template.html", data)
	})
	http.HandleFunc("/de/cards/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/de/cards/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		data, ok := deCardDataMap[cardNumber]
		if !ok {
			http.Error(w, "Card not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, "card-template.html", data)
	})

	// ########################
	// ########################
	// ########################

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
	
	// ########################
	
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
	
	// ########################
	
	http.HandleFunc("/en/index4", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "Information",
			H1Content:         "HTML & CSS - EN.html",
			PContent:          "Click the button below to toggle the visibility:",
			ButtonContent:     "Toggle Visibility",
			AddButton1Content: "Switch to 1nd File",
			AddButton2Content: "Switch to 3rd File",
		}
		renderHTML(w, r, "generic_index4.html", data)
	})
	http.HandleFunc("/de/index4", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "Information",
			H1Content:         "HTML & CSS - DE.html",
			PContent:          "Auf den Button klicken um die Sichtbarkeit zu ändern:",
			ButtonContent:     "Sichtbarkeit umschalten",
			AddButton1Content: "Wechsel zur 1. Datei",
			AddButton2Content: "Wechsel zur 3. Datei",
		}
		renderHTML(w, r, "generic_index4.html", data)
	})

	// ########################
	
	http.HandleFunc("/en/index5", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "Contact",
			H1Content:         "HTML & CSS - EN.html",
			PContent:          "Click the button below to toggle the visibility:",
			ButtonContent:     "Toggle Visibility",
			AddButton1Content: "Switch to 1nd File",
			AddButton2Content: "Switch to 3rd File",
		}
		renderHTML(w, r, "generic_index5.html", data)
	})
	http.HandleFunc("/de/index5", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "Kontakt",
			H1Content:         "HTML & CSS - DE.html",
			PContent:          "Auf den Button klicken um die Sichtbarkeit zu ändern:",
			ButtonContent:     "Sichtbarkeit umschalten",
			AddButton1Content: "Wechsel zur 1. Datei",
			AddButton2Content: "Wechsel zur 3. Datei",
		}
		renderHTML(w, r, "generic_index5.html", data)
	})

	// ########################
	// ########################
	// ########################

	http.HandleFunc("/en/card/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/en/card/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid blog number", http.StatusBadRequest)
			return
		}
		data, exists := enBlogDataMap[cardNumber]
		if !exists {
			http.Error(w, "Blog not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	http.HandleFunc("/de/card/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/de/card/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid blog number", http.StatusBadRequest)
			return
		}
		data, exists := deBlogDataMap[cardNumber]
		if !exists {
			http.Error(w, "Blog not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, "generic_index3.html", data)
	})

	// ########################
	// ########################
	// ########################
	
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

func renderHTML(w http.ResponseWriter, _ *http.Request, templateFile string, data interface{}) {
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
