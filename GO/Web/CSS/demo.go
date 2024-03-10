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

	log.Fatal(http.ListenAndServe(":8080", nil))
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
	case "/":
		http.Redirect(w, r, "/en/index/1", http.StatusMovedPermanently)
	default:
		http.NotFound(w, r)
	}
}

func handleMaxNumRequest(w http.ResponseWriter, _ *http.Request) {
	maxNumber := len(enCardDataSlice) - 1
	w.Header().Set("Content-Type", "text/plain")
	_, err := fmt.Fprintf(w, "%d", maxNumber)
	if err != nil {
		http.Error(w, "Error writing response", http.StatusInternalServerError)
		return
	}
}

func removeTrailingSlash(w http.ResponseWriter, r *http.Request) {
	newURL := strings.TrimSuffix(r.URL.Path, "/")
	http.Redirect(w, r, newURL, http.StatusMovedPermanently)
}

func handleIndexRequest(w http.ResponseWriter, r *http.Request) {
	var indexMap map[int]PageData
	var data PageData
	var ok bool
	language := r.PathValue("language")
	indexString := r.PathValue("index")
	indexInt, err := strconv.Atoi(indexString)
	if err != nil {
		http.Error(w, "Error converting Path", http.StatusInternalServerError)
		return
	}
	switch language {
	case "en":
		indexMap = enIndexMap
	case "de":
		indexMap = deIndexMap
	default:
		http.Error(w, "Error with Index Map Language", http.StatusInternalServerError)
		return
	}
	data, ok = indexMap[indexInt]
	if ok {
		templateFile := "generic_index" + indexString + ".html"
		renderHTML(w, r, templateFile, data)
	} else {
		http.Error(w, "Error accessing Index Map", http.StatusInternalServerError)
	}
}

func handleSingleCard(w http.ResponseWriter, r *http.Request) {
	var cardSlice []CardData
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
		cardSlice = enBlogDataSlice
	case "de":
		cardSlice = deBlogDataSlice
	default:
		http.Error(w, "Error with Blog Data Map Language", http.StatusInternalServerError)
		return
	}
	if cardNumber >= 0 && cardNumber < len(cardSlice) {
		data := cardSlice[cardNumber]
		renderHTML(w, r, "generic_index3.html", data)
	} else {
		http.Error(w, "Card number out of bounds", http.StatusNotFound)
	}
}

func handleAllCards(w http.ResponseWriter, r *http.Request) {
	var cardSlice []CardData
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
		cardSlice = enCardDataSlice
	case "de":
		cardSlice = deCardDataSlice
	default:
		http.Error(w, "Error with Card Data Map Language", http.StatusInternalServerError)
		return
	}
	if cardNumber >= 0 && cardNumber < len(cardSlice) {
		data := cardSlice[cardNumber]
		renderHTML(w, r, "card-template.html", data)
	} else {
		http.Error(w, "Card number out of bounds", http.StatusNotFound)
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
		return
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
	Id          uint
	Language    string
	Year        uint
	Month       uint
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

var enCardDataSlice = []CardData{
	{
		Id:          0,
		Year:        1970,
		Month:       1,
		Title:       "Unix epoch",
		Description: "Start of time",
	},
	{
		Id:          1,
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Title",
		Description: "Description for Card 1.",
		Get:         "../card/1",
		Blog:        "Read Card",
	},
	{
		Id:          2,
		Year:        2022,
		Month:       2,
		Title:       "Ipsum Card 2 Title",
		Description: "Description for Card 2.",
		Get:         "../card/2",
		Blog:        "Read Card",
	},
}

var deCardDataSlice = []CardData{
	{
		Id:          0,
		Year:        1970,
		Month:       1,
		Title:       "Unix epoch",
		Description: "Start of time",
	},
	{
		Id:          1,
		Year:        2022,
		Month:       1,
		Title:       "Lorem Card 1 Titel",
		Description: "Beschreibung für Card 1.",
		Get:         "../card/1",
		Blog:        "Card lesen",
	},
	{
		Id:          2,
		Year:        2022,
		Month:       2,
		Title:       "Ipsum Card 2 Titel",
		Description: "Beschreibung für Card 2.",
		Get:         "../card/2",
		Blog:        "Card lesen",
	},
}

var enBlogDataSlice = []CardData{
	{
		Id:          0,
		Year:        1970,
		Month:       1,
		Title:       "Unix epoch",
		Description: "Start of time",
	},
	{
		Id:          1,
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
	{
		Id:          2,
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

var deBlogDataSlice = []CardData{
	{
		Id:          0,
		Year:        1970,
		Month:       1,
		Title:       "Unix epoch",
		Description: "Start of time",
	},
	{
		Id:          1,
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
	{
		Id:          2,
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
