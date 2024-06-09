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

	var executablePath, directory, fileName, outputFileName string
	var err error
	var files []os.FileInfo
	var file os.FileInfo
	var fileContent ods.Odsfile
	var outputContent []string
	var buf bytes.Buffer
	var records [][]string

	executablePath, err = os.Executable()
	if err != nil {
		handleError(err, "")
		return
	}

	directory = filepath.Dir(executablePath)

	files, err = readDirectory(directory)
	if err != nil {
		handleError(err, directory)
		return
	}

	for _, file = range files {

		fileName = file.Name()
		if !strings.HasSuffix(fileName, ".ods") {
			continue
		}

		fileContent, err = ods.ReadODSFile(filepath.Join(directory, fileName))
		if err != nil {
			handleError(err, fileName)
			continue
		}

		outputContent = extractContent(fileContent.Sheets[1])

		err = writeContent(&buf, outputContent)
		if err != nil {
			handleError(err, fileName)
			return
		}

		records, err = readCSVContent(&buf)
		if err != nil {
			handleError(err, fileName)
			return
		}

		outputFileName = fileName[:len(fileName)-len(filepath.Ext(fileName))] + "-info.xlsx"

		err = saveAsXlsx(records, outputFileName)
		if err != nil {
			handleError(err, fileName)
		}
	}
}

func handleError(originalErr error, fileOrDirName string) {

	var timestamp, fileName, errorMessage string
	var line int
	var err error
	var f *os.File

	timestamp = time.Now().Format("2006-01-02_15-04-05")
	fileName = fmt.Sprintf("crash_%s.txt", timestamp)
	_, _, line, _ = runtime.Caller(1)
	err = zenity.Error("See crash.txt", zenity.Title("Error"), zenity.ErrorIcon)

	f, err = os.Create(fileName)
	if err != nil {
		fmt.Println(err)
	}

	errorMessage = fmt.Sprintf("Error: %v\nOccurred at line: %d\n", originalErr.Error(), line)
	if fileOrDirName != "" {
		errorMessage += fmt.Sprintf("Related file or directory: %s\n", fileOrDirName)
	}
	_, err = f.WriteString(errorMessage)
	if err != nil {
		fmt.Println(err)
	}

	defer func() {
		err = f.Close()
		if err != nil {
			fmt.Println(err)
		}
	}()
}

func readDirectory(directory string) ([]os.FileInfo, error) {

	var dir *os.File
	var err error

	dir, err = os.Open(directory)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = dir.Close()
		if err != nil {
			fmt.Println(err)
		}
	}()
	return dir.Readdir(-1)
}

func extractContent(sheet ods.Sheet) []string {

	var row ods.Row
	var rowString string
	var col int
	var cell ods.Cell
	var outputContent []string

	for _, row = range sheet.Rows {
		rowString = ""
		for col, cell = range row.Cells {
			if col >= 21 { // max column truncation
				break
			}
			rowString = rowString + cell.Text + ";" // semicolon delimiter
		}
		outputContent = append(outputContent, rowString)
	}
	return outputContent
}

func writeContent(buf *bytes.Buffer, content []string) error {

	var w *bufio.Writer
	var line string
	var err error

	w = bufio.NewWriter(buf)
	for _, line = range content {
		if _, err = fmt.Fprintln(w, line+"\r"); err != nil {
			return err
		}
	}
	return w.Flush()
}

func readCSVContent(buf *bytes.Buffer) ([][]string, error) {

	var r *csv.Reader

	r = csv.NewReader(buf)
	r.Comma = ';'
	return r.ReadAll()
}

func findUniqueAG(records [][]string) map[string][]int {

	var uniqueEntries map[string][]int
	var i int
	var record []string
	var entry string
	var ok bool

	uniqueEntries = make(map[string][]int)
	for i, record = range records {
		if i == 0 {
			continue
		}
		entry = record[20] // 20 = Artikelgruppe
		_, ok = uniqueEntries[entry]
		if !ok {
			uniqueEntries[entry] = make([]int, 0)
		}
		uniqueEntries[entry] = append(uniqueEntries[entry], i)
	}
	return uniqueEntries
}

func combineUniqueAG(records [][]string, uniqueAGs map[string][]int) [][]interface{} {

	var AG, value string
	var occurrences []int
	var uniqueValues map[string]struct{}
	var rowIndex int
	var uniqueValueArray []string
	var combinedArray [][]interface{}

	for AG, occurrences = range uniqueAGs {
		uniqueValues = make(map[string]struct{})
		for _, rowIndex = range occurrences {
			uniqueValues[records[rowIndex][8]] = struct{}{}
			// as there is no set data structure like in other languages
			// we use an empty struct {} as a placeholder value to represent a set-like behavior
		}

		uniqueValueArray = make([]string, 0, len(uniqueValues))
		for value = range uniqueValues {
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

func findUniqueAN(records [][]string) map[string][]int {

	var uniqueEntries map[string][]int
	var i int
	var record []string
	var entry string

	uniqueEntries = make(map[string][]int)

	for i, record = range records {
		if i == 0 {
			continue
		}
		if len(record) > 8 {
			entry = record[8] // 8 = Artikelnummer
			uniqueEntries[entry] = append(uniqueEntries[entry], i)
		}
	}
	return uniqueEntries
}

func saveAsXlsx(records [][]string, fileName string) error {

	var wb *xlsx.File
	var sheet *xlsx.Sheet
	var err error
	var greyStyle, whiteStyle, headerStyle, currentStyle *xlsx.Style
	var firstRowOfGroup map[int]int
	var groupSum map[int]float64
	var uniqueAGs, uniqueEntries map[string][]int
	var combinedArray [][]interface{}
	var i, rowIndex, firstEntry, sumColumn13Column15, j, k int
	var entry []interface{}
	var firstColumnValue, value, lastGroup, tenthColumnValue string
	var values []string
	var row *xlsx.Row
	var cell1, cell, emptyCell *xlsx.Cell
	var sum, avg float64
	var uniqueValues map[string]struct{}
	var uniqueValueArray []string

	wb = xlsx.NewFile()
	sheet, err = wb.AddSheet("Entries")
	if err != nil {
		return err
	}

	greyStyle = createStyle("FFDCDCDC")
	whiteStyle = createStyle("FFFFFFFF")
	headerStyle = createStyle("FFFFFFFF")
	headerStyle.Border.Bottom = "medium"

	addHeaderRow(sheet, headerStyle)

	firstRowOfGroup = make(map[int]int)
	groupSum = make(map[int]float64)

	uniqueAGs = findUniqueAG(records)
	combinedArray = combineUniqueAG(records, uniqueAGs)
	uniqueEntries = findUniqueAN(records)

	for i, entry = range combinedArray {

		if (i % 2) == 1 {
			currentStyle = greyStyle
		} else {
			currentStyle = whiteStyle
		}

		firstColumnValue = entry[0].(string)

		values = entry[1].([]string)
		for _, value = range values {

			row = sheet.AddRow()
			cell1 = row.AddCell()
			cell1.Value = firstColumnValue
			cell1.SetStyle(currentStyle)

			_, rowIndex = cell1.GetCoordinates()
			if len(cell1.Value) != 0 {
				firstRowOfGroup[i] = rowIndex
			}

			addCell(row, value, currentStyle)

			firstEntry = uniqueEntries[value][0]

			// 10 = Artikelbezeichnung
			if firstEntry < len(records) && 11 < len(records[firstEntry]) {
				tenthColumnValue = records[firstEntry][10]
			}
			addCell(row, tenthColumnValue, currentStyle)

			// 11 = Preis pro Einheit
			sum, err = calculateSum(uniqueEntries[value], records, 11)
			if err != nil {
				return err
			}
			avg = roundFloat(sum/float64(len(uniqueEntries[value])), 3)
			addCellWithFormat(row, avg, currentStyle, "0.00 €;-0.00 €")

			// 12 = Menge pro Einheit
			// 14 = Menge pro Artikel
			sumColumn13Column15 = calculateSumColumns(records, uniqueEntries[value], 12, 14)
			addCell(row, sumColumn13Column15, currentStyle)

			// 18 = Brutto-Rabattpreis
			sum, err = calculateSum(uniqueEntries[value], records, 18)
			if err != nil {
				return err
			}
			addCellWithFormat(row, roundFloat(sum, 3), currentStyle, "0.00 €;-0.00 €")

			groupSum[i] += sum
			firstColumnValue = "" // Clear after the first iteration to avoid repeating
		}
	}

	for k = 0; k < len(groupSum); k++ {

		if (k % 2) == 1 {
			currentStyle = greyStyle
		} else {
			currentStyle = whiteStyle
		}

		cell, _ = sheet.Cell(firstRowOfGroup[k], 6)
		cell.SetFloatWithFormat(groupSum[k], "0.00 €;-0.00 €")
		cell.SetStyle(currentStyle)

		if k < len(groupSum)-1 {

			for j = firstRowOfGroup[k]; j < firstRowOfGroup[k+1]; j++ {
				emptyCell, _ = sheet.Cell(j, 6)
				emptyCell.SetStyle(currentStyle)
			}

		} else {
			lastGroup = combinedArray[len(combinedArray)-1][0].(string)

			uniqueValues = make(map[string]struct{})
			for _, rowIndex = range uniqueAGs[lastGroup] {
				uniqueValues[records[rowIndex][8]] = struct{}{}
			}

			uniqueValueArray = make([]string, 0, len(uniqueValues))
			for value = range uniqueValues {
				uniqueValueArray = append(uniqueValueArray, value)
			}
			for j = firstRowOfGroup[k]; j < firstRowOfGroup[k]+(len(uniqueValueArray)-1); j++ {
				emptyCell, _ = sheet.Cell(j, 6)
				emptyCell.SetStyle(currentStyle)
			}

		}

	}

	emptyCell, _ = sheet.Cell(j, 6)
	emptyCell.SetStyle(currentStyle)

	setColumnWidths(sheet)

	err = wb.Save(fileName)
	if err != nil {
		return err
	}

	return nil
}

func createStyle(color string) *xlsx.Style {

	var fill *xlsx.Fill
	var style *xlsx.Style

	fill = xlsx.NewFill("solid", color, "FF000000")
	style = xlsx.NewStyle()
	style.Fill = *fill
	style.Font.Size = 11
	style.ApplyFill = true
	return style
}

func addHeaderRow(sheet *xlsx.Sheet, headerStyle *xlsx.Style) {

	var headerRow *xlsx.Row
	var headers []string
	var header string
	var headerCell *xlsx.Cell

	headerRow = sheet.AddRow()
	headers = []string{"Artikelgruppe", "Artikelnummer", "Artikelbezeichnung", "Durschnittspreis", "Gesamtmenge", "Brutto-Preis", "Gesamtpreis"}
	for _, header = range headers {
		headerCell = headerRow.AddCell()
		headerCell.Value = header
		headerCell.SetStyle(headerStyle)
	}
}

func addCell(row *xlsx.Row, value interface{}, style *xlsx.Style) {

	var cell *xlsx.Cell

	cell = row.AddCell()
	cell.SetValue(value)
	cell.SetStyle(style)
}

func addCellWithFormat(row *xlsx.Row, value float64, style *xlsx.Style, format string) {

	var cell *xlsx.Cell

	cell = row.AddCell()
	cell.SetFloatWithFormat(value, format)
	cell.SetStyle(style)
}

func calculateSum(entries []int, records [][]string, column int) (float64, error) {

	var sum, val float64
	var index int
	var floatStrWithComma, floatStrWithDot string
	var err error

	sum = 0.0
	for _, index = range entries {
		floatStrWithComma = records[index][column]
		floatStrWithDot = strings.Replace(floatStrWithComma, ",", ".", 1)
		val, err = strconv.ParseFloat(floatStrWithDot, 64)
		if err != nil {
			return 0, err
		}
		sum += val
	}
	return sum, nil
}

func roundFloat(val float64, precision uint) float64 {

	var ratio float64

	ratio = math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

func calculateSumColumns(records [][]string, indices []int, col1, col2 int) int {

	var sum, index, val1, val2 int
	var err1, err2 error

	sum = 0
	for _, index = range indices {
		if index < len(records) && col2 < len(records[index]) {
			val1, err1 = strconv.Atoi(records[index][col1])
			val2, err2 = strconv.Atoi(records[index][col2])
			if err1 == nil && err2 == nil {
				sum += val1 * val2
			}
		}
	}
	return sum
}

func setColumnWidths(sheet *xlsx.Sheet) {

	var widths []float64
	var i int
	var width float64
	var newCol *xlsx.Col

	widths = []float64{10, 11, 19.5, 12, 11, 9, 9.5}
	for i, width = range widths {
		newCol = xlsx.NewColForRange(i+1, i+1)
		newCol.SetWidth(width)
		sheet.SetColParameters(newCol)
	}
}
