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
	files, err := readDirectory(directory)
	if err != nil {
		handleError(err, directory)
		return
	}

	for _, file := range files {
		fileName := file.Name()
		if !strings.HasSuffix(fileName, ".ods") {
			continue
		}
		var fileContent ods.Odsfile
		fileContent, err = ods.ReadODSFile(directory + "/" + fileName)
		if err != nil {
			handleError(err, fileName)
			continue
		}

		sheet := fileContent.Sheets[1]
		outputContent := extractContent(sheet)

		var buf bytes.Buffer
		if err = writeContent(&buf, outputContent); err != nil {
			handleError(err, fileName)
			return
		}

		var records [][]string
		records, err = readCSVContent(&buf)
		if err != nil {
			handleError(err, fileName)
			return
		}

		uniqueAGs := findUniqueAG(records)
		combinedArray := combineUniqueAG(records, uniqueAGs)

		if err = saveResults(fileName, records, combinedArray, uniqueAGs); err != nil {
			handleError(err, fileName)
		}
	}
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

func readDirectory(directory string) ([]os.FileInfo, error) {
	dir, err := os.Open(directory)
	if err != nil {
		return nil, err
	}
	defer func(dir *os.File) {
		err = dir.Close()
		if err != nil {
		}
	}(dir)
	return dir.Readdir(-1)
}

func extractContent(sheet ods.Sheet) []string {
	const maxColumns = 21
	var outputContent []string
	for _, row := range sheet.Rows {
		rowString := ""
		for j, cell := range row.Cells {
			if j >= maxColumns {
				break
			}
			rowString = rowString + cell.Text + ";" // semicolon delimiter
		}
		outputContent = append(outputContent, rowString)
	}
	return outputContent
}

func writeContent(buf *bytes.Buffer, content []string) error {
	w := bufio.NewWriter(buf)
	for _, line := range content {
		if _, err := fmt.Fprintln(w, line+"\r"); err != nil {
			return err
		}
	}
	return w.Flush()
}

func readCSVContent(buf *bytes.Buffer) ([][]string, error) {
	reader := csv.NewReader(buf)
	reader.Comma = ';'
	return reader.ReadAll()
}

func findUniqueAG(records [][]string) map[string][]int {
	uniqueEntries := make(map[string][]int)
	for i, record := range records {
		if i == 0 {
			continue
		}
		entry := record[20] // 20 = Artikelgruppe
		if _, ok := uniqueEntries[entry]; !ok {
			uniqueEntries[entry] = make([]int, 0)
		}
		uniqueEntries[entry] = append(uniqueEntries[entry], i)
	}
	return uniqueEntries
}

func combineUniqueAG(records [][]string, uniqueAGs map[string][]int) [][]interface{} {
	var combinedArray [][]interface{}

	for AG, occurrences := range uniqueAGs {
		uniqueValues := make(map[string]struct{})
		for _, rowIndex := range occurrences {
			uniqueValues[records[rowIndex][8]] = struct{}{}
		}

		uniqueValueArray := make([]string, 0, len(uniqueValues))
		for value := range uniqueValues {
			uniqueValueArray = append(uniqueValueArray, value)
		}
		sort.Strings(uniqueValueArray)
		combinedArray = append(combinedArray, []interface{}{AG, uniqueValueArray})
	}

	sort.Slice(combinedArray, func(i, j int) bool {
		return combinedArray[i][0].(string) > combinedArray[j][0].(string)
	})

	return combinedArray
}

func saveResults(fileName string, records [][]string, combinedArray [][]interface{}, uniqueAGs map[string][]int) error {
	base := fileName[:len(fileName)-len(filepath.Ext(fileName))]
	uniqueEntries := findUniqueAN(records)

	return saveAsXlsx(base+"-info.xlsx", uniqueEntries, records, combinedArray, uniqueAGs)
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

	for i, entry := range combinedArray {

		currentStyle := selectStyle(i, greyStyle, whiteStyle) // Determine if the current set of rows should be grey or white

		firstColumnValue := entry[0].(string)

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

			uniqueValues := make(map[string]struct{})
			for _, rowIndex := range occurrences {
				uniqueValues[records[rowIndex][8]] = struct{}{}
			}

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

func selectStyle(index int, greyStyle, whiteStyle *xlsx.Style) *xlsx.Style {
	if (index % 2) == 1 {
		return greyStyle
	}
	return whiteStyle
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

func roundFloat(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
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

func setColumnWidths(sheet *xlsx.Sheet) {
	widths := []float64{10, 11, 19.5, 12, 11, 9, 9.5}
	for i, width := range widths {
		newCol := xlsx.NewColForRange(i+1, i+1)
		newCol.SetWidth(width)
		sheet.SetColParameters(newCol)
	}
}
