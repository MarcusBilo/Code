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

	var (
		exPath, directory, fileName, outputFileName string
		err                                         error
		files, outputContent                        []string
		fileContent                                 ods.Odsfile
		buf                                         bytes.Buffer
		records                                     [][]string
	)

	exPath, err = os.Executable()
	if err != nil {
		handleError(err, "")
		return
	}

	directory = filepath.Dir(exPath)

	files, err = getOdsFilesInDirectory(directory)
	if err != nil {
		handleError(err, directory)
		return
	}

	for _, fileName = range files {

		buf.Reset() // Reset the buffer before reusing it

		fileContent, err = ods.ReadODSFile(filepath.Join(directory, fileName))
		if err != nil {
			handleError(err, fileName)
			continue
		}

		outputContent = extractContent(fileContent.Sheets[1])

		err = writeContent(&buf, outputContent)
		if err != nil {
			handleError(err, fileName)
			continue
		}

		records, err = readCSVContent(&buf)
		if err != nil {
			handleError(err, fileName)
			continue
		}

		outputFileName = fileName[:len(fileName)-len(filepath.Ext(fileName))] + "-info.xlsx"

		err = saveAsXlsx(records, outputFileName)
		if err != nil {
			handleError(err, fileName)
			continue
		}
	}
}

func handleError(originalErr error, fileOrDirName string) {

	var (
		timestamp, fileName, errorMessage string
		line                              int
		err                               error
		f                                 *os.File
	)

	timestamp = time.Now().Format("2006-01-02_15-04-05")
	fileName = fmt.Sprintf("crash_%s.txt", timestamp)
	_, _, line, _ = runtime.Caller(1)
	err = zenity.Error("See crash.txt", zenity.Title("Error"), zenity.ErrorIcon)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "error displaying error: %v\n", err)
	}

	f, err = os.Create(fileName)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "error creating file: %v\n", err)
	}

	errorMessage = fmt.Sprintf("Error: %v\nOccurred at line: %d\n", originalErr.Error(), line)
	if fileOrDirName != "" {
		errorMessage += fmt.Sprintf("Related file or directory: %s\n", fileOrDirName)
	}

	_, err = f.WriteString(errorMessage)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "error writing string: %v\n", err)
	}

	defer surelyClose(f)

}

func surelyClose(f *os.File) {
	err := f.Close()
	if err != nil {
		panic(fmt.Sprintf("error closing file: %v", err))
	}
}

func getOdsFilesInDirectory(directory string) ([]string, error) {

	var (
		dir                      *os.File
		err                      error
		entries, filteredEntries []string
	)

	dir, err = os.Open(directory)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "error opening file: %v\n", err)
		return nil, err
	}

	defer surelyClose(dir)

	entries, err = dir.Readdirnames(-1)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "error reading dir: %v\n", err)
		return nil, err
	}

	for _, entry := range entries {
		if strings.HasSuffix(entry, ".ods") {
			filteredEntries = append(filteredEntries, entry)
		}
	}

	return filteredEntries, nil
}

func extractContent(sheet ods.Sheet) []string {

	var (
		row           ods.Row
		rowString     string
		col           int
		cell          ods.Cell
		outputContent []string
	)

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

	var (
		w    *bufio.Writer
		line string
		err  error
	)

	w = bufio.NewWriter(buf)
	for _, line = range content {
		_, err = fmt.Fprintln(w, line+"\r")
		if err != nil {
			return err
		}
	}
	return w.Flush()
}

func readCSVContent(buf *bytes.Buffer) ([][]string, error) {

	var (
		r *csv.Reader
	)

	r = csv.NewReader(buf)
	r.Comma = ';'
	return r.ReadAll()
}

func saveAsXlsx(records [][]string, fileName string) error {

	var (
		wb                                 *xlsx.File
		sheet                              *xlsx.Sheet
		err                                error
		headerStyle, greyStyle, whiteStyle *xlsx.Style
		firstRowOfGroup                    map[int]int
		groupSum                           map[int]float64
		uniqueAGs                          map[string][]int
		combinedArray                      [][]interface{}
		lastGroup, header                  string
		headerRow                          *xlsx.Row
		headers                            []string
		headerCell                         *xlsx.Cell
	)

	wb = xlsx.NewFile()
	sheet, err = wb.AddSheet("Entries")
	if err != nil {
		return err
	}

	headerStyle, greyStyle, whiteStyle = createStyle("FFFFFFFF"), createStyle("FFDCDCDC"), createStyle("FFFFFFFF")

	headerStyle.Border.Bottom = "medium"
	headerRow = sheet.AddRow()
	headers = []string{"Artikelgruppe", "Artikelnummer", "Artikelbezeichnung", "Durschnittspreis", "Gesamtmenge", "Brutto-Preis", "Gesamtpreis"}
	for _, header = range headers {
		headerCell = headerRow.AddCell()
		headerCell.Value = header
		headerCell.SetStyle(headerStyle)
	}

	firstRowOfGroup = make(map[int]int)
	groupSum = make(map[int]float64)
	uniqueAGs = findUniqueAG(records)
	combinedArray = combineUniqueAG(records, uniqueAGs)
	lastGroup = combinedArray[len(combinedArray)-1][0].(string)

	firstRowOfGroup, groupSum, err = populateSheet(sheet, records, firstRowOfGroup, groupSum, combinedArray, greyStyle, whiteStyle)
	if err != nil {
		return err
	}

	err = finalizeSheet(sheet, records, firstRowOfGroup, groupSum, lastGroup, uniqueAGs, greyStyle, whiteStyle)
	if err != nil {
		return err
	}

	return wb.Save(fileName)
}

func createStyle(color string) *xlsx.Style {

	var (
		fill  *xlsx.Fill
		style *xlsx.Style
	)

	fill = xlsx.NewFill("solid", color, "FF000000")
	style = xlsx.NewStyle()
	style.Fill = *fill
	style.Font.Size = 11
	style.ApplyFill = true
	return style
}

func populateSheet(sheet *xlsx.Sheet, records [][]string, firstRowOfGroup map[int]int, groupSum map[int]float64, combinedArray [][]interface{}, greyStyle, whiteStyle *xlsx.Style) (map[int]int, map[int]float64, error) {

	var (
		uniqueEntries                                map[string][]int
		currentStyle                                 *xlsx.Style
		i, rowIndex, firstEntry, sumColumn13Column15 int
		entry                                        []interface{}
		firstColumnValue, value, tenthColumnValue    string
		values                                       []string
		row                                          *xlsx.Row
		cell                                         *xlsx.Cell
		sum, avg                                     float64
		err                                          error
	)

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
			cell = row.AddCell()
			cell.Value = firstColumnValue
			cell.SetStyle(currentStyle)

			_, rowIndex = cell.GetCoordinates()
			if len(cell.Value) != 0 {
				firstRowOfGroup[i] = rowIndex
			}

			cell = row.AddCell()
			cell.SetValue(value)
			cell.SetStyle(currentStyle)

			firstEntry = uniqueEntries[value][0]

			if firstEntry < len(records) && 11 < len(records[firstEntry]) {
				tenthColumnValue = records[firstEntry][10]
			}

			cell = row.AddCell()
			cell.SetValue(tenthColumnValue)
			cell.SetStyle(currentStyle)

			sum, err = calculateSum(uniqueEntries[value], records, 11)
			if err != nil {
				return nil, nil, err
			}
			avg = sum / float64(len(uniqueEntries[value]))

			cell = row.AddCell()
			cell.SetFloatWithFormat(math.Round(avg*1000)/1000, "0.00 €;-0.00 €")
			cell.SetStyle(currentStyle)

			sumColumn13Column15, err = calculateSumColumns(records, uniqueEntries[value], 12, 14)
			if err != nil {
				return nil, nil, err
			}

			cell = row.AddCell()
			cell.SetValue(sumColumn13Column15)
			cell.SetStyle(currentStyle)

			sum, err = calculateSum(uniqueEntries[value], records, 18)
			if err != nil {
				return nil, nil, err
			}

			cell = row.AddCell()
			cell.SetFloatWithFormat(math.Round(sum*1000)/1000, "0.00 €;-0.00 €")
			cell.SetStyle(currentStyle)

			groupSum[i] += sum
			firstColumnValue = ""
		}
	}

	return firstRowOfGroup, groupSum, nil
}

func finalizeSheet(sheet *xlsx.Sheet, records [][]string, firstRowOfGroup map[int]int, groupSum map[int]float64, lastGroup string, uniqueAGs map[string][]int, greyStyle, whiteStyle *xlsx.Style) error {

	var (
		currentStyle      *xlsx.Style
		value             string
		uniqueValues      map[string]struct{}
		uniqueValueArray  []string
		rowIndex, j, k, i int
		cell, emptyCell   *xlsx.Cell
		widths            []float64
		width             float64
		newCol            *xlsx.Col
	)

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

	widths = []float64{10, 11, 19.5, 12, 11, 9, 9.5}
	for i, width = range widths {
		newCol = xlsx.NewColForRange(i+1, i+1)
		newCol.SetWidth(width)
		sheet.SetColParameters(newCol)
	}

	return nil
}

func findUniqueAG(records [][]string) map[string][]int {

	var (
		uniqueEntries map[string][]int
		i             int
		record        []string
		entry         string
		ok            bool
	)

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

	var (
		AG, value        string
		occurrences      []int
		uniqueValues     map[string]struct{}
		rowIndex         int
		uniqueValueArray []string
		combinedArray    [][]interface{}
	)

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

	var (
		uniqueEntries map[string][]int
		i             int
		record        []string
		entry         string
	)

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

func calculateSum(entries []int, records [][]string, column int) (float64, error) {

	var (
		sum, val                           float64
		index                              int
		floatStrWithComma, floatStrWithDot string
		err                                error
	)

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

func calculateSumColumns(records [][]string, indices []int, col1, col2 int) (int, error) {

	var (
		sum, index, val1, val2 int
		err                    error
	)

	sum = 0
	for _, index = range indices {
		if index < len(records) && col2 < len(records[index]) {
			val1, err = strconv.Atoi(records[index][col1])
			if err != nil {
				return 0, err
			}
			val2, err = strconv.Atoi(records[index][col2])
			if err != nil {
				return 0, err
			}
			sum += val1 * val2
		}
	}
	return sum, nil
}
