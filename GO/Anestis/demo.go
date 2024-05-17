package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	ods "github.com/LIJUCHACKO/ods2csv"
	"github.com/tealeg/xlsx/v3"
	"log"
	"sort"
	"strconv"
)

func main() {
	fileContent, err := ods.ReadODSFile("2024-4.ods")
	if err != nil {
		fmt.Println("Error reading ODS file:", err)
		return
	}
	sheet := fileContent.Sheets[1]
	outputcontent := []string{}
	for i, row := range sheet.Rows {
		rowString := ""
		for j, cell := range row.Cells {
			if i == 0 && j == len(row.Cells)-1 { // Check if it's the last cell in the first row
				continue // if so skip it, as "Charge" is always empty
			}
			rowString = rowString + cell.Text + ";" // Add semicolon delimiter
		}
		outputcontent = append(outputcontent, rowString)
	}

	// Remove the last semicolon from each string in outputcontent
	for i := range outputcontent {
		if len(outputcontent[i]) > 0 && outputcontent[i][len(outputcontent[i])-1] == ';' {
			outputcontent[i] = outputcontent[i][:len(outputcontent[i])-1]
		}
	}

	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	for _, line := range outputcontent {
		fmt.Fprintln(w, line+"\r")
	}
	w.Flush()

	// Now you have the CSV content in the buffer 'buf'.
	// You can read it line by line.

	reader := csv.NewReader(&buf)
	reader.Comma = ';'
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV content:", err)
		return
	}

	// -----------------------------------------------------------------------------------------------------------------

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

	err = saveAsXlsx("output.xlsx", uniqueEntries, records, combinedArray)
	if err != nil {
		log.Fatal(err)
	}

}

func findUniqueAN(records [][]string) map[string][]int {
	uniqueEntries := make(map[string][]int)

	for i, record := range records {
		if i == 0 {
			continue
		}
		if len(record) > 8 {
			entry := record[8]
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
		entry := record[20] // Column 20 = Artikelgruppe
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
				tenthColumnValue = records[firstEntry][10]
			} else {
				tenthColumnValue = "N/A" // Fallback value if the indices are out of range
			}

			cell3 := row.AddCell()
			cell3.Value = tenthColumnValue
			cell3.SetStyle(currentStyle)

			sum := 0.0
			count := len(uniqueEntries[value])
			for _, index := range uniqueEntries[value] {
				val, err := strconv.ParseFloat(records[index][11], 64)
				if err == nil {
					sum += val
				}
			}
			avgValue := "N/A"
			if count > 0 {
				avg := sum / float64(count)
				avgValue = strconv.FormatFloat(avg, 'f', 3, 64)
			}
			cell4 := row.AddCell()
			cell4.Value = avgValue
			cell4.SetStyle(currentStyle)

			// Calculate sum of multiplied values of column 13 and column 15 for each unique entry
			sumColumn13Column15 := 0.0
			for _, index := range uniqueEntries[value] {
				if index < len(records) && len(records[index]) > 14 {
					val13, err := strconv.ParseFloat(records[index][12], 64)
					val15, err := strconv.ParseFloat(records[index][14], 64)
					if err == nil {
						sumColumn13Column15 += val13 * val15
					}
				}
			}
			cell5 := row.AddCell()
			cell5.Value = strconv.FormatFloat(sumColumn13Column15, 'f', 0, 64)
			cell5.SetStyle(currentStyle)

			// Check in the "records"
			var sixteenthColumnValue string
			if firstEntry < len(records) && 17 < len(records[firstEntry]) {
				sixteenthColumnValue = records[firstEntry][16]
			} else {
				sixteenthColumnValue = "N/A" // Fallback value if the indices are out of range
			}

			cell6 := row.AddCell()
			cell6.Value = sixteenthColumnValue
			cell6.SetStyle(currentStyle)

			sum = 0.0
			for _, index := range uniqueEntries[value] {
				val, err := strconv.ParseFloat(records[index][18], 64)
				if err == nil {
					sum += val
				}
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

	for i := 0; i < len(groupSum); i++ {

		var currentStyle *xlsx.Style
		if (i % 2) == 1 {
			currentStyle = greyStyle
		} else {
			currentStyle = whiteStyle
		}

		cell, _ := sheet.Cell(firstRowOfGroup[i], 7)
		cell.SetValue(groupSum[i])
		cell.SetStyle(currentStyle)

		// Fill potential empty cells until the next value with the same style
		for j := firstRowOfGroup[i] + 1; j <= firstRowOfGroup[i+1]-1; j++ {
			emptyCell, _ := sheet.Cell(j, 7)
			emptyCell.SetStyle(currentStyle)
		}
	}

	err = wb.Save(fileName)
	if err != nil {
		return err
	}

	return nil
}
