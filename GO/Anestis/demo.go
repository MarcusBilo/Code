package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	ods "github.com/LIJUCHACKO/ods2csv"
	"github.com/ncruces/zenity"
	"github.com/tealeg/xlsx/v3"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

func main() {

	executablePath, err := os.Executable()
	if err != nil {
		handleError(err, "")
		return
	}

	directory := filepath.Dir(executablePath)
	dir, err := os.Open(directory)
	if err != nil {
		handleError(err, directory)
		return
	}

	defer func() {
		if err = dir.Close(); err != nil {
			handleError(err, directory)
			return
		}
	}()

	files, err := dir.Readdir(-1)
	if err != nil {
		handleError(err, directory)
		return
	}

	for _, file := range files {
		fileName := file.Name()
		if !strings.HasSuffix(fileName, ".ods") {
			continue
		}
		fileContent, err := ods.ReadODSFile(directory + "/" + fileName)
		if err != nil {
			handleError(err, fileName)
			continue
		}

		// ########################################################

		sheet := fileContent.Sheets[1]
		var outputContent []string
		for i, row := range sheet.Rows {
			rowString := ""
			for j, cell := range row.Cells {
				if i == 0 && j == len(row.Cells)-1 { // Check if it's the last cell in the first row
					continue // if so skip it, as "Charge" is always empty
					//
					// nur die letzte spalte entfernen sofern es eine header spalte mehr gibt als werte
					// und auch nur entfernen wenn die letzte spalte "Charge" ist ?
					// zumindest checken ob es nicht vllt doch einen wert in charge gibt
					// falls doch dann nicht löschen?
					//
				}
				rowString = rowString + cell.Text + ";" // Add semicolon delimiter
			}
			outputContent = append(outputContent, rowString)
		}

		// Remove the last semicolon from each string in outputContent
		// Ist mit der Charge änderung drin
		// muss sich ebenfalls angeschaut werden
		for i := range outputContent {
			if len(outputContent[i]) > 0 && outputContent[i][len(outputContent[i])-1] == ';' {
				outputContent[i] = outputContent[i][:len(outputContent[i])-1]
			}
		}

		// ########################################################

		var buf bytes.Buffer
		w := bufio.NewWriter(&buf)
		for _, line := range outputContent {
			_, err := fmt.Fprintln(w, line+"\r")
			if err != nil {
				handleError(err, fileName)
				break
			}
		}
		err = w.Flush()
		if err != nil {
			handleError(err, fileName)
			continue
		}

		// Now you have the CSV content in the buffer 'buf'.
		// You can read it line by line.

		reader := csv.NewReader(&buf)
		reader.Comma = ';'
		records, err := reader.ReadAll()
		if err != nil {
			handleError(err, fileName)
			continue
		}

		uniqueAGs := findUniqueAG(records)

		var combinedArray [][]interface{}

		for AG, occurrences := range uniqueAGs {
			uniqueValues := make(map[string]struct{})
			for _, rowIndex := range occurrences {
				value := records[rowIndex][8]
				uniqueValues[value] = struct{}{}
			}
			uniqueValueArray := make([]string, 0, len(uniqueValues))
			for value := range uniqueValues {
				uniqueValueArray = append(uniqueValueArray, value)
			}
			combinedArray = append(combinedArray, []interface{}{AG, uniqueValueArray})
		}

		// Sorting combinedArray based on AG
		sort.Slice(combinedArray, func(i, j int) bool {
			return combinedArray[i][0].(string) > combinedArray[j][0].(string)
		})

		for _, arr := range combinedArray {
			secondPart := arr[1].([]string)
			sort.Strings(secondPart)
		}

		uniqueEntries := findUniqueAN(records)

		base := fileName[:len(fileName)-len(filepath.Ext(fileName))]

		err = saveAsXlsx(base+"-info.xlsx", uniqueEntries, records, combinedArray, uniqueAGs)
		if err != nil {
			handleError(err, fileName)
			continue
		}
	}
}

func findUniqueAN(records [][]string) map[string][]int {
	uniqueEntries := make(map[string][]int)

	for i, record := range records {
		if i == 0 {
			continue
		}
		if len(record) > 8 {
			entry := record[8] // 8 = Artikelnummer
			uniqueEntries[entry] = append(uniqueEntries[entry], i)
		}
	}
	return uniqueEntries
}

func findUniqueAG(records [][]string) map[string][]int {
	uniqueEntries := make(map[string][]int)
	for i, record := range records {
		if i == 0 {
			continue
		}
		entry := record[20] // 20 = Artikelgruppe
		if _, ok := uniqueEntries[entry]; !ok {
			uniqueEntries[entry] = make([]int, 0) // Initialize the slice if it doesn't exist
		}
		uniqueEntries[entry] = append(uniqueEntries[entry], i)
	}
	return uniqueEntries
}

func saveAsXlsx(fileName string, uniqueEntries map[string][]int, records [][]string, combinedArray [][]interface{}, uniqueAGs map[string][]int) error {
	wb := xlsx.NewFile()
	sheet, err := wb.AddSheet("Entries")
	if err != nil {
		return err
	}

	greyStyle, whiteStyle, headerStyle := createStyles()

	addHeaderRow(sheet, headerStyle)

	groupSum := make(map[int]float64)
	firstRowOfGroup := make(map[int]int)

	// Iterate over all entries in combinedArray
	for i, entry := range combinedArray {
		// Determine if the current set of rows should be grey or white
		currentStyle := selectStyle(i, greyStyle, whiteStyle)

		// First column value
		firstColumnValue := entry[0].(string)

		// Iterate over the slice in the second element
		values := entry[1].([]string)
		for _, value := range values {

			row := sheet.AddRow()
			cell1 := row.AddCell()
			cell1.Value = firstColumnValue
			cell1.SetStyle(currentStyle)

			_, rowIndex := cell1.GetCoordinates()
			if len(cell1.Value) != 0 {
				firstRowOfGroup[i] = rowIndex
			}

			addCell(row, value, currentStyle)

			// Get the first entry in the list from uniqueEntries
			firstEntry := uniqueEntries[value][0]

			// 10 = Artikelbezeichnung
			var tenthColumnValue string
			if firstEntry < len(records) && 11 < len(records[firstEntry]) {
				tenthColumnValue = records[firstEntry][10]
			}

			addCell(row, tenthColumnValue, currentStyle)

			// 11 = Preis pro Einheit
			sum := 0.0
			count := len(uniqueEntries[value])
			for _, index := range uniqueEntries[value] {
				floatStrWithComma := records[index][11]
				floatStrWithDot := strings.Replace(floatStrWithComma, ",", ".", 1)
				val, err := strconv.ParseFloat(floatStrWithDot, 64)
				if err != nil {
					return err
				}
				sum += val
			}
			avg := roundFloat(sum/float64(count), 3)
			addCellWithFormat(row, avg, currentStyle, "0.00 €;-0.00 €")

			// 12 = Menge pro Einheit
			// 14 = Menge pro Artikel
			sumColumn13Column15 := calculateSumColumns(records, uniqueEntries[value], 12, 14)
			addCell(row, sumColumn13Column15, currentStyle)

			// 18 = Brutto-Rabattpreis
			sum = 0.0
			for _, index := range uniqueEntries[value] {
				floatStrWithComma := records[index][18]
				floatStrWithDot := strings.Replace(floatStrWithComma, ",", ".", 1)
				val, err := strconv.ParseFloat(floatStrWithDot, 64)
				if err != nil {
					return err
				}
				sum += val
			}
			sumValue := roundFloat(sum, 3)
			addCellWithFormat(row, sumValue, currentStyle, "0.00 €;-0.00 €")

			groupSum[i] += sum
			firstColumnValue = "" // Clear after the first iteration to avoid repeating
		}
	}

	var currentStyle *xlsx.Style
	var j int
	for i := 0; i < len(groupSum); i++ {

		currentStyle = selectStyle(i, greyStyle, whiteStyle)

		cell, _ := sheet.Cell(firstRowOfGroup[i], 6)
		cell.SetFloatWithFormat(groupSum[i], "0.00 €;-0.00 €")
		cell.SetStyle(currentStyle)

		if i < len(groupSum)-1 {

			for j = firstRowOfGroup[i]; j < firstRowOfGroup[i+1]; j++ {
				emptyCell, _ := sheet.Cell(j, 6)
				emptyCell.SetStyle(currentStyle)
			}

		} else {
			lastGroup := combinedArray[len(combinedArray)-1][0].(string)
			occurrences := uniqueAGs[lastGroup]

			// Create a map to store unique values
			uniqueValues := make(map[string]struct{})
			for _, rowIndex := range occurrences {
				uniqueValues[records[rowIndex][8]] = struct{}{}
			}

			// Convert the unique values map to an array
			uniqueValueArray := make([]string, 0, len(uniqueValues))
			for value := range uniqueValues {
				uniqueValueArray = append(uniqueValueArray, value)
			}
			for j = firstRowOfGroup[i]; j < firstRowOfGroup[i]+(len(uniqueValueArray)-1); j++ {
				emptyCell, _ := sheet.Cell(j, 6)
				emptyCell.SetStyle(currentStyle)
			}

		}

	}

	emptyCell, _ := sheet.Cell(j, 6)
	emptyCell.SetStyle(currentStyle)

	setColumnWidths(sheet)

	err = wb.Save(fileName)
	if err != nil {
		return err
	}

	return nil
}

func createStyles() (*xlsx.Style, *xlsx.Style, *xlsx.Style) {
	greyStyle := createStyle("FFDCDCDC")
	whiteStyle := createStyle("FFFFFFFF")
	headerStyle := createStyle("FFFFFFFF")
	headerStyle.Border.Bottom = "medium"
	return greyStyle, whiteStyle, headerStyle
}

func createStyle(color string) *xlsx.Style {
	fill := xlsx.NewFill("solid", color, "FF000000")
	style := xlsx.NewStyle()
	style.Fill = *fill
	style.Font.Size = 11
	style.ApplyFill = true
	return style
}

func addHeaderRow(sheet *xlsx.Sheet, headerStyle *xlsx.Style) {
	headerRow := sheet.AddRow()
	headers := []string{"Artikelgruppe", "Artikelnummer", "Artikelbezeichnung", "Durschnittspreis", "Gesamtmenge", "Brutto-Preis", "Gesamtpreis"}
	for _, header := range headers {
		headerCell := headerRow.AddCell()
		headerCell.Value = header
		headerCell.SetStyle(headerStyle)
	}
}

func setColumnWidths(sheet *xlsx.Sheet) {
	newCol := xlsx.NewColForRange(1, 1)
	newCol.SetWidth(8.5)
	sheet.SetColParameters(newCol)
	newCol = xlsx.NewColForRange(2, 7)
	newCol.SetWidth(12)
	sheet.SetColParameters(newCol)
}

func handleError(originalErr error, fileOrDirName string) {
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	fileName := fmt.Sprintf("crash_%s.txt", timestamp)
	_, _, line, _ := runtime.Caller(1)
	err := zenity.Error("See crash.txt", zenity.Title("Error"), zenity.ErrorIcon)

	f, err := os.Create(fileName)
	if err != nil {
		fmt.Println(err)
	}

	errorMessage := fmt.Sprintf("Error: %v\nOccurred at line: %d\n", originalErr.Error(), line)
	if fileOrDirName != "" {
		errorMessage += fmt.Sprintf("Related file or directory: %s\n", fileOrDirName)
	}
	_, err = f.WriteString(errorMessage)
	if err != nil {
		fmt.Println(err)
	}

	defer func() {
		if err = f.Close(); err != nil {
			fmt.Println(err)
		}
	}()
}

func roundFloat(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

func addCell(row *xlsx.Row, value interface{}, style *xlsx.Style) {
	cell := row.AddCell()
	cell.SetValue(value)
	cell.SetStyle(style)
}

func addCellWithFormat(row *xlsx.Row, value float64, style *xlsx.Style, format string) {
	cell := row.AddCell()
	cell.SetFloatWithFormat(value, format)
	cell.SetStyle(style)
}

func calculateSumColumns(records [][]string, indices []int, col1, col2 int) int {
	sum := 0
	for _, index := range indices {
		if index < len(records) && col2 < len(records[index]) {
			val1, err1 := strconv.Atoi(records[index][col1])
			val2, err2 := strconv.Atoi(records[index][col2])
			if err1 == nil && err2 == nil {
				sum += val1 * val2
			}
		}
	}
	return sum
}

func selectStyle(index int, greyStyle, whiteStyle *xlsx.Style) *xlsx.Style {
	if (index % 2) == 1 {
		return greyStyle
	}
	return whiteStyle
}
