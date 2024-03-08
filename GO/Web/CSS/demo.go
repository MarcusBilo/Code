package main

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"strconv"
	"strings"
)

// Go 1.22rc2

func main() {

	http.Handle("GET /", http.HandlerFunc(handleBaseRequest))
	http.Handle("GET /max-card-number", http.HandlerFunc(handleMaxNumRequest))
	http.Handle("GET /{language}/index/{index}/", http.HandlerFunc(removeTrailingSlash))
	http.Handle("GET /{language}/index/{index}", http.HandlerFunc(handleIndexRequest))
	http.Handle("GET /{language}/cards/", http.HandlerFunc(handleAllCards))
	http.Handle("GET /{language}/card/", http.HandlerFunc(handleSingleCard))

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}

func handleBaseRequest(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/styles.css":
		w.Header().Set("Content-Type", "text/css")
		http.ServeFile(w, r, "styles.css")
	case "/htmx_v1.9.10.min.js":
		// https://unpkg.com/browse/htmx.org@1.9.10/dist/
		w.Header().Set("Content-Type", "application/javascript")
		http.ServeFile(w, r, "htmx_v1.9.10.min.js")
	case "/Noto-Sans-regular.woff2":
		// https://github.com/pages-themes/minimal/blob/master/assets/fonts/Noto-Sans-regular/Noto-Sans-regular.woff2
		http.ServeFile(w, r, "Noto-Sans-regular.woff2")
	default:
		http.Redirect(w, r, "/en/index/1", http.StatusMovedPermanently)
	}
}

func handleMaxNumRequest(w http.ResponseWriter, r *http.Request) {
	maxNumber := len(enCardDataMap)
	w.Header().Set("Content-Type", "text/plain")
	_, err := fmt.Fprintf(w, "%d", maxNumber)
	if err != nil {
		http.Error(w, "Error writing response", http.StatusInternalServerError)
		return
	}
}

func removeTrailingSlash(w http.ResponseWriter, r *http.Request) {
	if strings.HasSuffix(r.URL.Path, "/") {
		newURL := strings.TrimSuffix(r.URL.Path, "/")
		http.Redirect(w, r, newURL, http.StatusMovedPermanently)
		return
	}
}

func handleIndexRequest(w http.ResponseWriter, r *http.Request) {
	var data PageData
	var ok bool
	language := r.PathValue("language")
	indexString := r.PathValue("index")
	indexInt, err := strconv.Atoi(indexString)
	if err != nil {
		http.Error(w, "Error converting Path", http.StatusInternalServerError)
	}
	switch language {
	case "en":
		data, ok = enIndexMap[indexInt]
	case "de":
		data, ok = deIndexMap[indexInt]
	default:
		http.Error(w, "Error with Index Map Language", http.StatusInternalServerError)
		return
	}
	if !ok {
		http.Error(w, "Error accessing Index Map", http.StatusInternalServerError)
		return
	}
	htmlTemplate := "generic_index" + indexString + ".html"
	renderHTML(w, r, htmlTemplate, data)
}

func handleSingleCard(w http.ResponseWriter, r *http.Request) {
	var data CardData
	var ok bool
	var cardNumber int
	language := r.PathValue("language")
	format := "/" + language + "/card/%d"
	_, err := fmt.Sscanf(r.URL.Path, format, &cardNumber)
	if err != nil {
		http.Error(w, "Invalid card number", http.StatusInternalServerError)
		return
	}
	switch language {
	case "en":
		data, ok = enBlogDataMap[cardNumber]
	case "de":
		data, ok = deBlogDataMap[cardNumber]
	default:
		http.Error(w, "Error with Blog Data Map Language", http.StatusInternalServerError)
		return
	}
	if !ok {
		http.Error(w, "Error accessing Blog Data Map", http.StatusInternalServerError)
		return
	}
	renderHTML(w, r, "generic_index3.html", data)
}

func handleAllCards(w http.ResponseWriter, r *http.Request) {
	var data CardData
	var ok bool
	var cardNumber int
	language := r.PathValue("language")
	format := "/" + language + "/cards/%d"
	_, err := fmt.Sscanf(r.URL.Path, format, &cardNumber)
	if err != nil {
		http.Error(w, "Invalid cards number", http.StatusInternalServerError)
		return
	}
	switch language {
	case "en":
		data, ok = enCardDataMap[cardNumber]
	case "de":
		data, ok = deCardDataMap[cardNumber]
	default:
		http.Error(w, "Error with Card Data Map Language", http.StatusInternalServerError)
		return
	}
	if !ok {
		http.Error(w, "Error accessing Card Data Map", http.StatusInternalServerError)
		return
	}
	renderHTML(w, r, "card-template.html", data)
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
