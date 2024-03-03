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
	PContent1         string
	PContent2         string
	ButtonContent     string
	AddButton1Content string
	AddButton2Content string
}

type CardData struct {
	Language    string
	Year        int
	Month       int
	Title       string
	Description string
	Get         string
	Blog        string
	H1Content   string
	PContent1   string
	PContent2   string
	PContent3   string
	PContent4   string
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

var enBlogDataMap = map[int]CardData{
	1: {
		Language:    "en",
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Title",
		Description: "Description Card 1.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
	},
	2: {
		Language:    "en",
		Year:        2022,
		Month:       2,
		Title:       "Lorem Card 2 Title",
		Description: "Description Card 2.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
	},
}

var deBlogDataMap = map[int]CardData{
	1: {
		Language:    "de",
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Titel",
		Description: "Beschreibung Card 1.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
	},
	2: {
		Language:    "de",
		Year:        2022,
		Month:       2,
		Title:       "Lorem Card 2 Titel",
		Description: "Beschreibung Card 2.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
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
			Language:  "en",
			Title:     "HTMX & Basic CSS",
			H1Content: "HTMX & Basic CSS - Index1EN.html",
			PContent1: "Landing Page with filler content",
		}
		renderHTML(w, r, "generic_index1.html", data)
	})
	http.HandleFunc("/de/index1", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:  "de",
			Title:     "HTMX & Basic CSS",
			H1Content: "HTMX & Basic CSS - Index1DE.html",
			PContent1: "Startseite mit Füllinhalten",
		}
		renderHTML(w, r, "generic_index1.html", data)
	})

	// ########################

	http.HandleFunc("/en/index2", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language: "en",
			Title:    "HTMX & Basic CSS",
		}
		renderHTML(w, r, "generic_index2.html", data)
	})
	http.HandleFunc("/de/index2", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language: "de",
			Title:    "HTMX & Basic CSS",
		}
		renderHTML(w, r, "generic_index2.html", data)
	})

	// ########################

	http.HandleFunc("/en/index4", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:  "en",
			Title:     "Information",
			H1Content: "Exampleweb",
			PContent1: "Main Street 123",
			PContent2: "Zip Code: 12345",
		}
		renderHTML(w, r, "generic_index4.html", data)
	})
	http.HandleFunc("/de/index4", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:  "de",
			Title:     "Information",
			H1Content: "Exampleweb",
			PContent1: "Hauptstraße 123",
			PContent2: "Postleitzahl: 12345",
		}
		renderHTML(w, r, "generic_index4.html", data)
	})

	// ########################

	http.HandleFunc("/en/index5", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:  "en",
			Title:     "Contact",
			H1Content: "Send us an e-mail to",
			PContent1: "support@exampleweb.com",
			PContent2: "or write to us on social media",
		}
		renderHTML(w, r, "generic_index5.html", data)
	})
	http.HandleFunc("/de/index5", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:  "de",
			Title:     "Kontakt",
			H1Content: "Senden Sie uns eine E-Mail an",
			PContent1: "support@exampleweb.com",
			PContent2: "oder schreiben Sie uns auf Social Media an",
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
