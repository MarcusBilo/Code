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

var globalIndexMap = make(map[string]PageData)
var globalBlogDataMap = make(map[string]CardData)
var globalCardDataMap = make(map[string]CardData)
var globalCardDataMapLength = make(map[string]int)
var mutex sync.Mutex

func prepareMaps() {
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
		time.Sleep(300 * time.Second)
	}
}

func main() {

	go prepareMaps()

	/*
		func() {
			var wg sync.WaitGroup
			wg.Add(1)
			start := time.Now().UnixMicro()
			go func() {
				defer wg.Done()
				prepareMaps()
			}()
			wg.Wait()
			end := time.Now().UnixMicro()
			fmt.Println(end-start, "μs")
		}()
	*/

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
	// start := time.Now().UnixMicro()
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
	// end := time.Now().UnixMicro()
	// fmt.Printf("handleIndexRequest took %d microseconds\n", end-start)
}

func handleSingleCard(w http.ResponseWriter, r *http.Request) {
	// start := time.Now().UnixMicro()
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
	// end := time.Now().UnixMicro()
	// fmt.Printf("handleSingleCard took %d microseconds\n", end-start)
}

func handleAllCards(w http.ResponseWriter, r *http.Request) {
	// start := time.Now().UnixMicro()
	language := r.PathValue("language")
	mutex.Lock()
	for i := globalCardDataMapLength[language]; i >= 1; i-- {
		card, ok := globalCardDataMap[language+strconv.Itoa(i)]
		if !ok {
			http.Error(w, getLineAndTime(), http.StatusInternalServerError)
		}
		renderCardTemplate(w, r, "card-template.html", card)
	}
	defer mutex.Unlock()
	// end := time.Now().UnixMicro()
	// fmt.Printf("handleAllCards took %d microseconds\n", end-start)
}

func renderCardTemplate(w http.ResponseWriter, _ *http.Request, templateFile string, data interface{}) {
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

	var compressedPureHTML string
	compressedPureHTML = strings.Join(strings.Fields(buf.String()), " ")

	_, err = w.Write([]byte(compressedPureHTML))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func renderHTML(w http.ResponseWriter, r *http.Request, templateFile string, data interface{}) {
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
