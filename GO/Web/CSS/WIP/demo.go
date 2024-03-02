package main

import (
	"encoding/json"
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

type Card struct {
	Year        int    `json:"year"`
	Month       string `json:"month"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

func main() {

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/htmx_v1.9.10.min.js":
			// https://unpkg.com/browse/htmx.org@1.9.10/dist/
			w.Header().Set("Content-Type", "application/javascript")
			http.ServeFile(w, r, "htmx_v1.9.10.min.js")
		case "/styles.css":
			w.Header().Set("Content-Type", "text/css")
			http.ServeFile(w, r, "styles.css")
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
			maxNumber := getMaxCardNumber()
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

	http.HandleFunc("/de/cards/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/de/cards/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		cardData, err := fetchDeCardData(cardNumber)
		if err != nil {
			http.Error(w, "Error fetching card data", http.StatusInternalServerError)
			return
		}
		jsonData, err := json.Marshal(cardData)
		if err != nil {
			http.Error(w, "Error converting to JSON", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, err = w.Write(jsonData)
		if err != nil {
			return
		}
	})

	http.HandleFunc("/en/cards/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/en/cards/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		cardData, err := fetchEnCardData(cardNumber)
		if err != nil {
			http.Error(w, "Error fetching card data", http.StatusInternalServerError)
			return
		}
		jsonData, err := json.Marshal(cardData)
		if err != nil {
			http.Error(w, "Error converting to JSON", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, err = w.Write(jsonData)
		if err != nil {
			return
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
	http.HandleFunc("/en/card/1", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "HTMX & Basic CSS Example",
			AddButton1Content: "Back to 1st File",
			AddButton2Content: "Back to 2nd File",
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	http.HandleFunc("/de/card/1", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "HTMX & Basic CSS Beispiel",
			AddButton1Content: "Zurück zur 1. Datei",
			AddButton2Content: "Zurück zur 2. Datei",
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	http.HandleFunc("/en/card/2", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "HTMX & Basic CSS 2 Example",
			AddButton1Content: "Back to 1st File",
			AddButton2Content: "Back to 2nd File",
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	http.HandleFunc("/de/card/2", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "HTMX & Basic CSS 2 Beispiel",
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

func renderHTML(w http.ResponseWriter, _ *http.Request, templateFile string, data PageData) {
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

func fetchDeCardData(cardNumber int) (*Card, error) {
	switch cardNumber {
	case 1:
		return &Card{
			Year:        2022,
			Month:       "01",
			Title:       "Lorem ipsum",
			Description: "Dies ist die erste Kartenbeschreibung. Dies ist die erste Kartenbeschreibung. Dies ist die erste Kartenbeschreibung.",
		}, nil
	case 2:
		return &Card{
			Year:        2022,
			Month:       "02",
			Title:       "dolor sit amet",
			Description: "Dies ist die zweite Kartenbeschreibung.",
		}, nil
	default:
		return nil, fmt.Errorf("DE Card not found")
	}
}

func fetchEnCardData(cardNumber int) (*Card, error) {
	switch cardNumber {
	case 1:
		return &Card{
			Year:        2022,
			Month:       "01",
			Title:       "Lorem ipsum",
			Description: "This is the first card description. This is the first card description. This is the first card description.",
		}, nil
	case 2:
		return &Card{
			Year:        2022,
			Month:       "02",
			Title:       "dolor sit amet",
			Description: "This is the second card description.",
		}, nil
	default:
		return nil, fmt.Errorf("EN Card not found")
	}
}

func getMaxCardNumber() int {
	return 2
}
