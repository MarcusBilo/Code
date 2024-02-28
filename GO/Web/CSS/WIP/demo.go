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

	http.HandleFunc("/cards/", func(w http.ResponseWriter, r *http.Request) {
		var cardNumber int
		_, err := fmt.Sscanf(r.URL.Path, "/cards/%d", &cardNumber)
		if err != nil {
			http.Error(w, "Invalid card number", http.StatusBadRequest)
			return
		}
		cardData, err := fetchCardData(cardNumber)
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

	http.HandleFunc("/max-card-number", func(w http.ResponseWriter, r *http.Request) {
		maxNumber := getMaxCardNumber()
		w.Header().Set("Content-Type", "text/plain")
		_, err := fmt.Fprintf(w, "%d", maxNumber)
		if err != nil {
			http.Error(w, "Error writing response", http.StatusInternalServerError)
			return
		}
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/htmx_v1.9.10.min.js":
			w.Header().Set("Content-Type", "application/javascript")
			http.ServeFile(w, r, "htmx_v1.9.10.min.js")
		case "/styles.css":
			w.Header().Set("Content-Type", "text/css")
			http.ServeFile(w, r, "styles.css")
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
		default:
			http.Redirect(w, r, "/en/index1", http.StatusMovedPermanently)
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
	http.HandleFunc("/en/index3", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "en",
			Title:             "HTMX & Basic CSS Example",
			AddButton1Content: "Back to 1st File",
			AddButton2Content: "Back to 2nd File",
		}
		renderHTML(w, r, "generic_index3.html", data)
	})
	http.HandleFunc("/de/index3", func(w http.ResponseWriter, r *http.Request) {
		data := PageData{
			Language:          "de",
			Title:             "HTMX & Basic CSS Beispiel",
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

func renderHTML(w http.ResponseWriter, r *http.Request, templateFile string, data PageData) {
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

func fetchCardData(cardNumber int) (*Card, error) {
	// Here you would fetch the card data based on the card number from your data source
	// For simplicity, let's create two dummy cards with different data for card numbers 1 and 2
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
		return nil, fmt.Errorf("Card not found")
	}
}

func getMaxCardNumber() int {
	// Logic to determine the maximum card number dynamically
	return 2
}
