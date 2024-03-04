package main

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
)

func main() {

	http.HandleFunc("/", handleBaseRequest())
	http.HandleFunc("/max-card-number", handleMaxNumRequest())
	http.HandleFunc("/en/index/1", handleIndexRequest(enIndexMap, "generic_index1.html", "en"))
	http.HandleFunc("/de/index/1", handleIndexRequest(deIndexMap, "generic_index1.html", "de"))
	http.HandleFunc("/en/index/2", handleIndexRequest(enIndexMap, "generic_index2.html", "en"))
	http.HandleFunc("/de/index/2", handleIndexRequest(deIndexMap, "generic_index2.html", "de"))
	http.HandleFunc("/en/index/4", handleIndexRequest(enIndexMap, "generic_index4.html", "en"))
	http.HandleFunc("/de/index/4", handleIndexRequest(deIndexMap, "generic_index4.html", "de"))
	http.HandleFunc("/en/index/5", handleIndexRequest(enIndexMap, "generic_index5.html", "en"))
	http.HandleFunc("/de/index/5", handleIndexRequest(deIndexMap, "generic_index5.html", "de"))
	http.HandleFunc("/en/cards/", handleCardsRequest(enCardDataMap, "en"))
	http.HandleFunc("/de/cards/", handleCardsRequest(deCardDataMap, "de"))
	http.HandleFunc("/en/card/", handleCardRequest(enBlogDataMap, "en"))
	http.HandleFunc("/de/card/", handleCardRequest(deBlogDataMap, "de"))

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}

func handleBaseRequest() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
		case "/de.svg":
			// https://flagicons.lipis.dev/
			http.ServeFile(w, r, "de.svg")
		case "/gb.svg":
			// https://flagicons.lipis.dev/
			http.ServeFile(w, r, "gb.svg")
		default:
			http.Redirect(w, r, "/en/index/1", http.StatusMovedPermanently)
		}
	}
}

func handleMaxNumRequest() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		maxNumber := len(enCardDataMap)
		w.Header().Set("Content-Type", "text/plain")
		_, err := fmt.Fprintf(w, "%d", maxNumber)
		if err != nil {
			http.Error(w, "Error writing response", http.StatusInternalServerError)
			return
		}
	}
}

func handleIndexRequest(indexMap map[int]PageData, template string, lang string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var indexNumber int
		format := "/" + lang + "/index/%d"
		_, err := fmt.Sscanf(r.URL.Path, format, &indexNumber)
		if err != nil {
			http.Error(w, "Invalid index number", http.StatusBadRequest)
			return
		}
		data, ok := indexMap[indexNumber]
		if !ok {
			http.Error(w, "Index not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, template, data)
	}
}

func handleCardRequest(blogDataMap map[int]CardData, lang string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		format := "/" + lang + "/card/%d"
		_, err := fmt.Sscanf(r.URL.Path, format, &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		data, ok := blogDataMap[cardNumber]
		if !ok {
			http.Error(w, "Card not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, "generic_index3.html", data)
	}
}

func handleCardsRequest(cardDataMap map[int]CardData, lang string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		format := "/" + lang + "/cards/%d"
		_, err := fmt.Sscanf(r.URL.Path, format, &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		data, ok := cardDataMap[cardNumber]
		if !ok {
			http.Error(w, "Card not found", http.StatusNotFound)
			return
		}
		renderHTML(w, r, "card-template.html", data)
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

// ###############################################################################
// ################################## Mock Data ##################################
// ###############################################################################

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
		Get:         "../card/1",
		Blog:        "Read Card",
	},
	2: {
		Year:        2022,
		Month:       2,
		Title:       "Ipsum Card 2 Title",
		Description: "Description for Card 2.",
		Get:         "../card/2",
		Blog:        "Read Card",
	},
}

var deCardDataMap = map[int]CardData{
	1: {
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Titel",
		Description: "Beschreibung für Card 1.",
		Get:         "../card/1",
		Blog:        "Card lesen",
	},
	2: {
		Year:        2022,
		Month:       2,
		Title:       "Ipsum Card 2 Titel",
		Description: "Beschreibung für Card 2.",
		Get:         "../card/2",
		Blog:        "Card lesen",
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

var enIndexMap = map[int]PageData{
	1: {
		Language:  "en",
		Title:     "HTMX & Basic CSS",
		H1Content: "HTMX & Basic CSS - Index1EN.html",
		PContent1: "Landing Page with filler content",
	},
	2: {
		Language: "en",
		Title:    "HTMX & Basic CSS",
	},
	4: {
		Language:  "en",
		Title:     "Information",
		H1Content: "Exampleweb",
		PContent1: "Main Street 123",
		PContent2: "Zip Code: 12345",
	},
	5: {
		Language:  "en",
		Title:     "Contact",
		H1Content: "Send us an e-mail to",
		PContent1: "support@exampleweb.com",
		PContent2: "or write to us on social media",
	},
}

var deIndexMap = map[int]PageData{
	1: {
		Language:  "de",
		Title:     "HTMX & Basic CSS",
		H1Content: "HTMX & Basic CSS - Index1DE.html",
		PContent1: "Startseite mit Füllinhalten",
	},
	2: {
		Language: "de",
		Title:    "HTMX & Basic CSS",
	},
	4: {
		Language:  "de",
		Title:     "Information",
		H1Content: "Exampleweb",
		PContent1: "Hauptstraße 123",
		PContent2: "Postleitzahl: 12345",
	},
	5: {
		Language:  "de",
		Title:     "Kontakt",
		H1Content: "Senden Sie uns eine E-Mail an",
		PContent1: "support@exampleweb.com",
		PContent2: "oder schreiben Sie uns auf Social Media an",
	},
}
