package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	ods "github.com/LIJUCHACKO/ods2csv"
	"github.com/ncruces/zenity"
	"github.com/tealeg/xlsx/v3"
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
				}
				rowString = rowString + cell.Text + ";" // Add semicolon delimiter
			}
			outputContent = append(outputContent, rowString)
		}

		// Remove the last semicolon from each string in outputContent
		for i := range outputContent {
			if len(outputContent[i]) > 0 && outputContent[i][len(outputContent[i])-1] == ';' {
				outputContent[i] = outputContent[i][:len(outputContent[i])-1]
			}
		}

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

		// -------------------------------------------------------------------------------------------------------------

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

		err = saveAsXlsx(base+"-info.xlsx", uniqueEntries, records, combinedArray)
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

func saveAsXlsx(fileName string, uniqueEntries map[string][]int, records [][]string, combinedArray [][]interface{}) error {
	wb := xlsx.NewFile()
	sheet, err := wb.AddSheet("Entries")
	if err != nil {
		return err
	}

	greyFill := xlsx.NewFill("solid", "FFDCDCDC", "FF000000")
	greyStyle := xlsx.NewStyle()
	greyStyle.Fill = *greyFill
	greyStyle.ApplyFill = true

	whiteFill := xlsx.NewFill("solid", "FFFFFFFF", "FF000000")
	whiteStyle := xlsx.NewStyle()
	whiteStyle.Fill = *whiteFill
	whiteStyle.ApplyFill = true

	headerFill := xlsx.NewFill("solid", "FFFFFFFF", "FF000000")
	headerStyle := xlsx.NewStyle()
	headerStyle.Border.Bottom = "medium"
	headerStyle.Fill = *headerFill
	headerStyle.ApplyFill = true

	// Add header row
	headerRow := sheet.AddRow()
	headers := []string{"Artikelgruppe", "Artikelnummer", "Artikelbezeichnung", "Durschnittspreis", "Gesamtmenge", "Steuern", "Brutto-Preis", "Gesamtpreis"}
	for _, header := range headers {
		headerCell := headerRow.AddCell()
		headerCell.Value = header
		headerCell.SetStyle(headerStyle)
	}

	groupSum := make(map[int]float64)
	firstRowOfGroup := make(map[int]int)

	// Iterate over all entries in combinedArray
	for i, entry := range combinedArray {
		// Determine if the current set of rows should be grey or white
		var currentStyle *xlsx.Style
		if (i % 2) == 1 {
			currentStyle = greyStyle
		} else {
			currentStyle = whiteStyle
		}

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

			cell2 := row.AddCell()
			cell2.Value = value
			cell2.SetStyle(currentStyle)

			// Get the first entry in the list from uniqueEntries
			firstEntry := uniqueEntries[value][0]

			// Check in the "records"
			var tenthColumnValue string
			if firstEntry < len(records) && 11 < len(records[firstEntry]) {
				tenthColumnValue = records[firstEntry][10] // 10 = Artikelbezeichnung
			} else {
				tenthColumnValue = "N/A" // Fallback value if the indices are out of range
			}

			cell3 := row.AddCell()
			cell3.Value = tenthColumnValue
			cell3.SetStyle(currentStyle)

			sum := 0.0
			count := len(uniqueEntries[value])
			for _, index := range uniqueEntries[value] {
				val, err := strconv.ParseFloat(records[index][11], 64) // 11 = Preis pro Einheit
				if err != nil {
					return err
				}
				sum += val
			}
			avgValue := "N/A"
			if count > 0 {
				avg := sum / float64(count)
				avgValue = strconv.FormatFloat(avg, 'f', 3, 64)
			}
			cell4 := row.AddCell()
			cell4.Value = avgValue
			cell4.SetStyle(currentStyle)

			// Calculate sum of multiplied values for each unique entry
			sumColumn13Column15 := 0.0
			for _, index := range uniqueEntries[value] {
				if index < len(records) && len(records[index]) > 14 {
					val13, err := strconv.ParseFloat(records[index][12], 64) // 12 = Menge pro Einheit
					if err != nil {
						return err
					}
					val15, err := strconv.ParseFloat(records[index][14], 64) // 14 = Menge pro Artikel
					if err != nil {
						return err
					}
					sumColumn13Column15 += val13 * val15
				}
			}
			cell5 := row.AddCell()
			cell5.Value = strconv.FormatFloat(sumColumn13Column15, 'f', 0, 64)
			cell5.SetStyle(currentStyle)

			// Check in the "records"
			var sixteenthColumnValue string
			if firstEntry < len(records) && 17 < len(records[firstEntry]) {
				sixteenthColumnValue = records[firstEntry][16] // 16 = Steuer
			} else {
				sixteenthColumnValue = "N/A" // Fallback value if the indices are out of range
			}

			cell6 := row.AddCell()
			cell6.Value = sixteenthColumnValue
			cell6.SetStyle(currentStyle)

			sum = 0.0
			for _, index := range uniqueEntries[value] {
				val, err := strconv.ParseFloat(records[index][18], 64) // 18 = Brutto-Rabattpreis
				if err != nil {
					return err
				}
				sum += val
			}
			avgValue = "N/A"
			sumValue := strconv.FormatFloat(sum, 'f', 2, 64)
			cell7 := row.AddCell()
			cell7.Value = sumValue
			cell7.SetStyle(currentStyle)

			groupSum[i] += sum

			firstColumnValue = "" // Clear after the first iteration to avoid repeating
		}
	}

	var currentStyle *xlsx.Style
	var j int
	for i := 0; i < len(groupSum); i++ {

		if (i % 2) == 1 {
			currentStyle = greyStyle
		} else {
			currentStyle = whiteStyle
		}

		cell, _ := sheet.Cell(firstRowOfGroup[i], 7)
		cell.SetValue(groupSum[i])
		cell.SetStyle(currentStyle)

		// Fill potential empty cells with the same style
		for j = firstRowOfGroup[i] + 1; j <= firstRowOfGroup[i+1]; j++ {
			emptyCell, _ := sheet.Cell(j, 7)
			emptyCell.SetStyle(currentStyle)
		}
	}
	emptyCell, _ := sheet.Cell(j, 7)
	emptyCell.SetStyle(currentStyle)

	err = wb.Save(fileName)
	if err != nil {
		return err
	}

	return nil
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
