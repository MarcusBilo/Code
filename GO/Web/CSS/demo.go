package main

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Go 1.22rc2

var (
	indexCache           map[string]PageData
	indexCacheMutex      sync.Mutex
	lastIndexCacheUpdate time.Time
)

func main() {
	indexCache = make(map[string]PageData)

	http.Handle("/", onlyHandleGET(http.HandlerFunc(handleBaseRequest)))
	http.Handle("/{language}/index/{index}/", onlyHandleGET(http.HandlerFunc(removeTrailingSlash)))
	http.Handle("/{language}/index/{index}", onlyHandleGET(http.HandlerFunc(handleIndexRequest)))
	http.Handle("/{language}/all-cards", onlyHandleGET(http.HandlerFunc(handleAllCards)))
	http.Handle("/{language}/card/", onlyHandleGET(http.HandlerFunc(handleSingleCard)))

	log.Fatal(http.ListenAndServe(":8080", nil))
}

func onlyHandleGET(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, getLineAndTime(), http.StatusNotImplemented)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func handleBaseRequest(w http.ResponseWriter, r *http.Request) {
	fs := http.FileServer(http.Dir("."))
	switch r.URL.Path {
	case "/styles.css":
		w.Header().Set("Content-Type", "text/css")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/htmx_v1.9.10.min.js":
		// https://unpkg.com/browse/htmx.org@1.9.10/dist/
		w.Header().Set("Content-Type", "text/javascript")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/Noto-Sans-regular.woff2":
		// https://github.com/pages-themes/minimal/blob/master/assets/fonts/Noto-Sans-regular/Noto-Sans-regular.woff2
		w.Header().Set("Content-Type", "font/woff2")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/":
		http.Redirect(w, r, "/en/index/1", http.StatusMovedPermanently)
	default:
		http.NotFound(w, r)
	}
}

func removeTrailingSlash(w http.ResponseWriter, r *http.Request) {
	newURL := strings.TrimSuffix(r.URL.Path, "/")
	http.Redirect(w, r, newURL, http.StatusMovedPermanently)
}

func handleIndexRequest(w http.ResponseWriter, r *http.Request) {
	language := r.PathValue("language")
	index := r.PathValue("index")
	indexCacheMutex.Lock()
	defer indexCacheMutex.Unlock()
	data, exist := indexCache[language+"_"+index]
	if exist && time.Since(lastIndexCacheUpdate) < time.Minute {
		templateFile := "generic_index" + index + ".html"
		renderHTML(w, r, templateFile, data)
	} else {
		cacheAndRenderIndexData(language, index, w, r)
	}
}

func cacheAndRenderIndexData(language string, index string, w http.ResponseWriter, r *http.Request) {
	var indexMap map[string]PageData
	switch language {
	case "en":
		indexMap = enIndexMap
	case "de":
		indexMap = deIndexMap
	default:
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
		return
	}
	data, ok := indexMap[index]
	if ok {
		indexCache[language+"_"+index] = data
		lastIndexCacheUpdate = time.Now()
		templateFile := "generic_index" + index + ".html"
		renderHTML(w, r, templateFile, data)
	} else {
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
	}
}

func handleSingleCard(w http.ResponseWriter, r *http.Request) {
	var cardSlice []CardData
	var cardNumber int
	language := r.PathValue("language")
	format := "/" + language + "/card/%d"
	_, err := fmt.Sscanf(r.URL.Path, format, &cardNumber)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	switch language {
	case "en":
		cardSlice = enBlogDataSlice
	case "de":
		cardSlice = deBlogDataSlice
	default:
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
		return
	}
	cardNumber = cardNumber - 1
	if cardNumber >= 0 && cardNumber < len(cardSlice) {
		data := cardSlice[cardNumber]
		renderHTML(w, r, "generic_index3.html", data)
	} else {
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
		return
	}
}

func handleAllCards(w http.ResponseWriter, r *http.Request) {
	var cardSlice []CardData
	language := r.PathValue("language")
	switch language {
	case "en":
		cardSlice = enCardDataSlice
	case "de":
		cardSlice = deCardDataSlice
	default:
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
		return
	}
	maxIndex := len(cardSlice) - 1
	for i := maxIndex; i >= 0; i-- {
		data := cardSlice[i]
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
		return
	}
}

func getLineAndTime() string {
	_, _, line, _ := runtime.Caller(1)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	return fmt.Sprintf("%s | Line: %d", timestamp, line)
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
	Id          uint8
	Year        uint8
	Month       uint8
	Language    string
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
		Id:          1,
		Year:        22,
		Month:       1,
		Title:       "Lorem Card 1 Title",
		Description: "Description for Card 1.",
		Get:         "../card/1",
		Blog:        "Read Card",
	},
	{
		Id:          2,
		Year:        22,
		Month:       2,
		Title:       "Ipsum Card 2 Title",
		Description: "Description for Card 2.",
		Get:         "../card/2",
		Blog:        "Read Card",
	},
}

var deCardDataSlice = []CardData{
	{
		Id:          1,
		Year:        22,
		Month:       1,
		Title:       "Lorem Card 1 Titel",
		Description: "Beschreibung für Card 1.",
		Get:         "../card/1",
		Blog:        "Card lesen",
	},
	{
		Id:          2,
		Year:        22,
		Month:       2,
		Title:       "Ipsum Card 2 Titel",
		Description: "Beschreibung für Card 2.",
		Get:         "../card/2",
		Blog:        "Card lesen",
	},
}

var enBlogDataSlice = []CardData{
	{
		Id:          1,
		Year:        22,
		Month:       1,
		Language:    "en",
		Title:       "Lorem Card 1 Title",
		Description: "Description Card 1.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
	},
	{
		Id:          2,
		Year:        22,
		Month:       2,
		Language:    "en",
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
		Id:          1,
		Year:        22,
		Month:       1,
		Language:    "de",
		Title:       "Lorem Card 1 Titel",
		Description: "Beschreibung Card 1.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
	},
	{
		Id:          2,
		Year:        22,
		Month:       2,
		Language:    "de",
		Title:       "Lorem Card 2 Titel",
		Description: "Beschreibung Card 2.",
		PContent1:   "text1",
		PContent2:   "text2",
		PContent3:   "text3",
		PContent4:   "text4",
	},
}

var enIndexMap = map[string]PageData{
	"1": {
		Language:  "en",
		Title:     "HTMX & Basic CSS",
		H1Content: "HTMX & Basic CSS - Index1EN.html",
		PContent1: "Landing Page with filler content",
	},
	"2": {
		Language: "en",
		Title:    "HTMX & Basic CSS",
	},
	"4": {
		Language:  "en",
		Title:     "Information",
		H1Content: "Exampleweb",
		PContent1: "Main Street 123",
		PContent2: "Zip Code: 12345",
	},
	"5": {
		Language:  "en",
		Title:     "Contact",
		H1Content: "Send us an e-mail to",
		PContent1: "support@exampleweb.com",
		PContent2: "or write to us on social media",
	},
}

var deIndexMap = map[string]PageData{
	"1": {
		Language:  "de",
		Title:     "HTMX & Basic CSS",
		H1Content: "HTMX & Basic CSS - Index1DE.html",
		PContent1: "Startseite mit Füllinhalten",
	},
	"2": {
		Language: "de",
		Title:    "HTMX & Basic CSS",
	},
	"4": {
		Language:  "de",
		Title:     "Information",
		H1Content: "Exampleweb",
		PContent1: "Hauptstraße 123",
		PContent2: "Postleitzahl: 12345",
	},
	"5": {
		Language:  "de",
		Title:     "Kontakt",
		H1Content: "Senden Sie uns eine E-Mail an",
		PContent1: "support@exampleweb.com",
		PContent2: "oder schreiben Sie uns auf Social Media an",
	},
}
