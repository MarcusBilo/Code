package main

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Go 1.23.0

var globalIndexMap = make(map[string]IndexData, 9)
var globalBlogDataMap = make(map[string]BlogData, 13)
var globalCardDataMap = make(map[string]CardData, 13)
var globalCardDataMapLength = make(map[string]int, 3)
var mutex sync.Mutex

func prepareMaps() {

	/*
		fmt.Println(len(globalIndexMap))
		fmt.Println(len(globalBlogDataMap))
		fmt.Println(len(globalCardDataMap))
		fmt.Println(len(globalCardDataMapLength))
	*/

	for {
		func() {
			mutex.Lock()
			for index, data := range enIndexMap {
				globalIndexMap["en"+index] = data
			}
			for index, data := range deIndexMap {
				globalIndexMap["de"+index] = data
			}
			defer mutex.Unlock()
		}()
		func() {
			mutex.Lock()
			for index, data := range enBlogDataSlice {
				globalBlogDataMap["en"+strconv.Itoa(index+1)] = data
			}
			for index, data := range deBlogDataSlice {
				globalBlogDataMap["de"+strconv.Itoa(index+1)] = data
			}
			defer mutex.Unlock()
		}()
		func() {
			mutex.Lock()
			for index, data := range enCardDataSlice {
				globalCardDataMap["en"+strconv.Itoa(index+1)] = data
			}
			for index, data := range deCardDataSlice {
				globalCardDataMap["de"+strconv.Itoa(index+1)] = data
			}
			defer mutex.Unlock()
		}()
		func() {
			mutex.Lock()
			index := 1
			for {
				key := fmt.Sprintf("%s%d", "en", index)
				if _, ok := globalCardDataMap[key]; !ok {
					break
				}
				index++
			}
			globalCardDataMapLength["en"] = index - 1
			defer mutex.Unlock()
		}()
		func() {
			mutex.Lock()
			index := 1
			for {
				key := fmt.Sprintf("%s%d", "de", index)
				if _, ok := globalCardDataMap[key]; !ok {
					break
				}
				index++
			}
			globalCardDataMapLength["de"] = index - 1
			defer mutex.Unlock()
		}()

		/*
			fmt.Println(len(globalIndexMap))
			fmt.Println(len(globalBlogDataMap))
			fmt.Println(len(globalCardDataMap))
			fmt.Println(len(globalCardDataMapLength))
		*/

		time.Sleep(300 * time.Second)
	}
}

func main() {

	go prepareMaps()

	http.Handle("/", onlyHandleGET(http.HandlerFunc(handleBaseRequest)))
	http.Handle("/{language}/index/{index}/", onlyHandleGET(http.HandlerFunc(removeTrailingSlash)))
	http.Handle("/{language}/index/{index}", onlyHandleGET(http.HandlerFunc(handleIndexRequest)))
	http.Handle("/{language}/all-cards", onlyHandleGET(http.HandlerFunc(handleAllCards)))
	http.Handle("/{language}/card/{card}/", onlyHandleGET(http.HandlerFunc(removeTrailingSlash)))
	http.Handle("/{language}/card/{card}", onlyHandleGET(http.HandlerFunc(handleSingleCard)))

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
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			r.URL.Path += ".gz"
			w.Header().Set("Content-Encoding", "gzip")
		}
		w.Header().Set("Content-Type", "text/css")
		w.Header().Set("Cache-Control", "public, max-age=3600")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/htmx_v1.9.12.min.js":
		// https://unpkg.com/browse/htmx.org@1.9.12/dist/
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			r.URL.Path += ".gz"
			w.Header().Set("Content-Encoding", "gzip")
		}
		w.Header().Set("Content-Type", "text/javascript")
		w.Header().Set("Cache-Control", "public, max-age=3600")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/htmx_preload.js":
		// https://unpkg.com/browse/htmx.org@1.9.12/dist/ext/
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			r.URL.Path += ".gz"
			w.Header().Set("Content-Encoding", "gzip")
		}
		w.Header().Set("Content-Type", "text/javascript")
		w.Header().Set("Cache-Control", "public, max-age=3600")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/Noto-Sans-regular.woff2":
		// gzip not possible
		// https://github.com/pages-themes/minimal/blob/master/assets/fonts/Noto-Sans-regular/Noto-Sans-regular.woff2
		w.Header().Set("Content-Type", "font/woff2")
		w.Header().Set("Cache-Control", "public, max-age=3600")
		http.StripPrefix("/", fs).ServeHTTP(w, r)
	case "/arial-boldmt-webfont.woff2":
		// gzip not possible
		// https://www.fontsquirrel.com/tools/webfont-generator
		w.Header().Set("Content-Type", "font/woff2")
		w.Header().Set("Cache-Control", "public, max-age=3600")
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
	mutex.Lock()
	data, ok := globalIndexMap[language+index]
	defer mutex.Unlock()
	if !ok {
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
	}
	templateFile := "generic_index" + index + ".html"
	renderHTML(w, r, templateFile, data)
}

func handleSingleCard(w http.ResponseWriter, r *http.Request) {
	var cardNumber int
	language := r.PathValue("language")
	format := "/" + language + "/card/%d"
	_, err := fmt.Sscanf(r.URL.Path, format, &cardNumber)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	mutex.Lock()
	data, ok := globalBlogDataMap[language+strconv.Itoa(cardNumber)]
	defer mutex.Unlock()
	if !ok {
		http.Error(w, getLineAndTime(), http.StatusInternalServerError)
	}
	renderHTML(w, r, "generic_index3.html", data)
}

func handleAllCards(w http.ResponseWriter, r *http.Request) {
	language := r.PathValue("language")

	mutex.Lock()
	numberOfCards := globalCardDataMapLength[language]
	cards := make([]interface{}, 0, numberOfCards)
	for i := numberOfCards; i >= 1; i-- {
		card, ok := globalCardDataMap[language+strconv.Itoa(i)]
		if !ok {
			http.Error(w, getLineAndTime(), http.StatusInternalServerError)
			return
		}
		cards = append(cards, card)
	}
	defer mutex.Unlock()

	renderCardTemplate(w, r, "card-template.html", cards)
}

func renderCardTemplate(w http.ResponseWriter, _ *http.Request, templateFile string, data []interface{}) {
	// start := time.Now().UnixMicro()
	w.Header().Set("Content-Type", "text/html")
	w.Header().Set("Cache-Control", "public, max-age=300")
	tmpl, err := template.ParseFiles(templateFile)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var buf bytes.Buffer
	err = tmpl.Execute(&buf, data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	compressedPureHTML := strings.Join(strings.Fields(buf.String()), " ")

	_, err = w.Write([]byte(compressedPureHTML))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// end := time.Now().UnixMicro()
	// fmt.Printf("renderCardTemplate took %d microseconds\n", end-start)
}

func renderHTML(w http.ResponseWriter, r *http.Request, templateFile string, data interface{}) {
	// start := time.Now().UnixMicro()
	w.Header().Set("Content-Type", "text/html")
	w.Header().Set("Cache-Control", "public, max-age=300")

	tmpl, err := template.ParseFiles(templateFile)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")

		gzipWriter := gzip.NewWriter(w)
		defer surelyClose(gzipWriter)

		err = tmpl.Execute(gzipWriter, data)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	} else {
		err = tmpl.Execute(w, data)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	// end := time.Now().UnixMicro()
	// fmt.Printf("renderHTML took %d microseconds\n", end-start)
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

// #####################################################################################################################
// ################################## Mock Data ########################################################################
// #####################################################################################################################

type IndexData struct {
	Language  string
	Title     string
	H1Content string
	PContent1 string
	PContent2 string
}

type CardData struct {
	Id          uint8
	Year        string
	Month       string
	Title       string
	Description string
	Get         string
	Blog        string
}

type BlogData struct {
	Id          uint8
	Year        uint8
	Month       uint8
	Language    string
	Title       string
	H1Content   string
	Description string
	PContent1   string
	PContent2   string
	PContent3   string
	PContent4   string
}

func NewEnCardData(id uint8, year, month, title, description string) CardData {
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

func NewDeCardData(id uint8, year, month, title, description string) CardData {
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

// https://leagueoflegends.fandom.com/wiki/List_of_champions

var enCardDataSlice = []CardData{
	NewEnCardData(1, "09", "02", "Alistar the Minotaur", "Melee, Mana, Vanguard"),
	NewEnCardData(2, "09", "02", "Annie the Dark Child", "Ranged, Mana, Burst"),
	NewEnCardData(3, "09", "02", "Ashe the Frost Archer", "Ranged, Mana, Marksman"),
	NewEnCardData(4, "09", "02", "Fiddlesticks the Ancient Fear", "Ranged, Mana, Specialist"),
	NewEnCardData(5, "09", "02", "Jax Grandmaster at Arms", "Melee, Mana, Skirmisher"),
	NewEnCardData(6, "09", "02", "Kayle the Righteous", "Melee, Mana, Specialist"),
}

var deCardDataSlice = []CardData{
	NewDeCardData(1, "09", "02", "Alistar, der Minotaurus", "Nahkampf, Mana, Vorkämpfer"),
	NewDeCardData(2, "09", "02", "Annie, das Kind der Finsternis", "Fernkampf, Mana, Burst"),
	NewDeCardData(3, "09", "02", "Ashe, die Frostbogenschützin", "Fernkampf, Mana, Schütze"),
	NewDeCardData(4, "09", "02", "Fiddlesticks, das uralte Unheil", "Fernkampf, Mana, Spezialist"),
	NewDeCardData(5, "09", "02", "Jax, der Großmeister der Waffen", "Nahkampf, Mana, Plänkler"),
	NewDeCardData(6, "09", "02", "Kayle, die Rechtschaffende", "Nahkampf, Mana, Spezialist"),
}

func NewBlogData(id, year, month uint8, language, title, description, pcontent1, pcontent2, pcontent3, pcontent4 string) BlogData {
	return BlogData{
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

// empty H1Content

var enBlogDataSlice = []BlogData{
	NewBlogData(1, 22, 1, "en", "Alistar", "Extended, full description Card 1.", "text1", "text2", "text3", "text4"),
	NewBlogData(2, 22, 2, "en", "Annie", "Extended, full description Card 2.", "text1", "text2", "text3", "text4"),
	NewBlogData(3, 22, 3, "en", "Ashe", "Extended, full description Card 3.", "text1", "text2", "text3", "text4"),
	NewBlogData(4, 22, 4, "en", "Fiddlesticks", "Extended, full description Card 4.", "text1", "text2", "text3", "text4"),
	NewBlogData(5, 22, 5, "en", "Jax", "Extended, full description Card 5.", "text1", "text2", "text3", "text4"),
	NewBlogData(6, 22, 6, "en", "Kayle", "Extended, full description Card 6.", "text1", "text2", "text3", "text4"),
}

var deBlogDataSlice = []BlogData{
	NewBlogData(1, 22, 1, "de", "Alistar", "Erweiterte, volle Beschreibung Card 1.", "text1", "text2", "text3", "text4"),
	NewBlogData(2, 22, 2, "de", "Annie", "Erweiterte, volle Beschreibung Card 2.", "text1", "text2", "text3", "text4"),
	NewBlogData(3, 22, 3, "de", "Ashe", "Erweiterte, volle Beschreibung Card 3.", "text1", "text2", "text3", "text4"),
	NewBlogData(4, 22, 4, "de", "Fiddlesticks", "Erweiterte, volle Beschreibung Card 4.", "text1", "text2", "text3", "text4"),
	NewBlogData(5, 22, 5, "de", "Jax", "Erweiterte, volle Beschreibung Card 5.", "text1", "text2", "text3", "text4"),
	NewBlogData(6, 22, 6, "de", "Kayle", "Erweiterte, volle Beschreibung Card 6.", "text1", "text2", "text3", "text4"),
}

var enIndexMap = map[string]IndexData{
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

var deIndexMap = map[string]IndexData{
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
