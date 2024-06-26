package main

import (
	"compress/gzip"
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
	case "/arial-boldmt-webfont.woff2":
		// https://www.fontsquirrel.com/tools/webfont-generator
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
		renderGzipHTML(w, r, templateFile, data)
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
		renderGzipHTML(w, r, templateFile, data)
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
		renderGzipHTML(w, r, "generic_index3.html", data)
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
		// dont gzip
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

func renderGzipHTML(w http.ResponseWriter, r *http.Request, templateFile string, data interface{}) {
	w.Header().Set("Content-Type", "text/html")
	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")

		gzipWriter := gzip.NewWriter(w)
		defer surelyClose(gzipWriter)

		tmpl, err := template.ParseFiles(templateFile)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		err = tmpl.Execute(gzipWriter, data)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	} else {
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
}

func surelyClose(w *gzip.Writer) {
	err := w.Close()
	if err != nil {
		panic(fmt.Sprintf("error closing writer: %v", err))
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

func NewEnCardData(id, year, month uint8, title, description string) CardData {
	return CardData{
		Id:          id,
		Year:        year,
		Month:       month,
		Title:       title,
		Description: description,
		Get:         fmt.Sprintf("../card/%d", id),
		Blog:        "Read Card",
	}
}

func NewDeCardData(id, year, month uint8, title, description string) CardData {
	return CardData{
		Id:          id,
		Year:        year,
		Month:       month,
		Title:       title,
		Description: description,
		Get:         fmt.Sprintf("../card/%d", id),
		Blog:        "Card lesen",
	}
}

// https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number#Generation_I
// https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_category

var enCardDataSlice = []CardData{
	NewEnCardData(1, 22, 1, "Bulbasaur", "Seed, dual-type Grass-Poison, Pokémon"),
	NewEnCardData(2, 22, 2, "Ivysaur", "Seed, dual-type Grass-Poison, Pokémon"),
	NewEnCardData(3, 22, 3, "Venusaur", "Seed, dual-type Grass-Poison, Pokémon"),
	NewEnCardData(4, 22, 4, "Charmander", "Lizard, mono-type Fire, Pokémon"),
	NewEnCardData(5, 22, 5, "Charmeleon", "Flame, mono-type Fire, Pokémon"),
	NewEnCardData(6, 22, 6, "Charizard", "Flame, dual-type Fire-Flying, Pokémon"),
}

var deCardDataSlice = []CardData{
	NewDeCardData(1, 22, 1, "Lorem Card 1 Titel", "Beschreibung für Card 1."),
	NewDeCardData(2, 22, 2, "Ipsum Card 2 Titel", "Beschreibung für Card 2."),
}

func NewBlogData(id, year, month uint8, language, title, description, pcontent1, pcontent2, pcontent3, pcontent4 string) CardData {
	return CardData{
		Id:          id,
		Year:        year,
		Month:       month,
		Language:    language,
		Title:       title,
		Description: description,
		PContent1:   pcontent1,
		PContent2:   pcontent2,
		PContent3:   pcontent3,
		PContent4:   pcontent4,
	}
}

var enBlogDataSlice = []CardData{
	NewBlogData(1, 22, 1, "en", "Lorem Card 1 Title", "Extended, full description Card 1.", "text1", "text2", "text3", "text4"),
	NewBlogData(2, 22, 2, "en", "Lorem Card 2 Title", "Extended, full description Card 2.", "text1", "text2", "text3", "text4"),
	NewBlogData(3, 22, 3, "en", "Lorem Card 3 Title", "Extended, full description Card 3.", "text1", "text2", "text3", "text4"),
	NewBlogData(4, 22, 4, "en", "Lorem Card 4 Title", "Extended, full description Card 4.", "text1", "text2", "text3", "text4"),
	NewBlogData(5, 22, 5, "en", "Lorem Card 5 Title", "Extended, full description Card 5.", "text1", "text2", "text3", "text4"),
	NewBlogData(6, 22, 6, "en", "Lorem Card 6 Title", "Extended, full description Card 6.", "text1", "text2", "text3", "text4"),
}

var deBlogDataSlice = []CardData{
	NewBlogData(1, 22, 1, "de", "Lorem Card 1 Titel", "Erweiterte, volle Beschreibung Card 1.", "text1", "text2", "text3", "text4"),
	NewBlogData(2, 22, 2, "de", "Lorem Card 2 Titel", "Erweiterte, volle Beschreibung Card 2.", "text1", "text2", "text3", "text4"),
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
