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
		files                                       []os.FileInfo
		file                                        os.FileInfo
		fileContent                                 ods.Odsfile
		outputContent                               []string
		buf                                         bytes.Buffer
		records                                     [][]string
	)

	exPath, err = os.Executable()
	if err != nil {
		handleError(err, "")
		return
	}

	directory = filepath.Dir(exPath)

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
		fmt.Println(err)
	}

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

	var (
		dir *os.File
		err error
	)

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
		lastGroup                          string
	)

	wb = xlsx.NewFile()
	sheet, err = wb.AddSheet("Entries")
	if err != nil {
		return err
	}

	headerStyle = createStyle("FFFFFFFF")
	headerStyle.Border.Bottom = "medium"
	addHeaderRow(sheet, headerStyle)

	firstRowOfGroup = make(map[int]int)
	groupSum = make(map[int]float64)
	uniqueAGs = findUniqueAG(records)
	combinedArray = combineUniqueAG(records, uniqueAGs)
	lastGroup = combinedArray[len(combinedArray)-1][0].(string)
	greyStyle, whiteStyle = createStyle("FFDCDCDC"), createStyle("FFFFFFFF")

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

func addHeaderRow(sheet *xlsx.Sheet, headerStyle *xlsx.Style) {

	var (
		headerRow  *xlsx.Row
		headers    []string
		header     string
		headerCell *xlsx.Cell
	)

	headerRow = sheet.AddRow()
	headers = []string{"Artikelgruppe", "Artikelnummer", "Artikelbezeichnung", "Durschnittspreis", "Gesamtmenge", "Brutto-Preis", "Gesamtpreis"}
	for _, header = range headers {
		headerCell = headerRow.AddCell()
		headerCell.Value = header
		headerCell.SetStyle(headerStyle)
	}
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
		cell1                                        *xlsx.Cell
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
			cell1 = row.AddCell()
			cell1.Value = firstColumnValue
			cell1.SetStyle(currentStyle)

			_, rowIndex = cell1.GetCoordinates()
			if len(cell1.Value) != 0 {
				firstRowOfGroup[i] = rowIndex
			}

			addCell(row, value, currentStyle)

			firstEntry = uniqueEntries[value][0]

			if firstEntry < len(records) && 11 < len(records[firstEntry]) {
				tenthColumnValue = records[firstEntry][10]
			}
			addCell(row, tenthColumnValue, currentStyle)

			sum, err = calculateSum(uniqueEntries[value], records, 11)
			if err != nil {
				return nil, nil, err
			}
			avg = roundFloat(sum/float64(len(uniqueEntries[value])), 3)
			addCellWithFormat(row, avg, currentStyle, "0.00 €;-0.00 €")

			sumColumn13Column15, err = calculateSumColumns(records, uniqueEntries[value], 12, 14)
			if err != nil {
				return nil, nil, err
			}
			addCell(row, sumColumn13Column15, currentStyle)

			sum, err = calculateSum(uniqueEntries[value], records, 18)
			if err != nil {
				return nil, nil, err
			}
			addCellWithFormat(row, roundFloat(sum, 3), currentStyle, "0.00 €;-0.00 €")

			groupSum[i] += sum
			firstColumnValue = ""
		}
	}

	return firstRowOfGroup, groupSum, nil
}

func finalizeSheet(sheet *xlsx.Sheet, records [][]string, firstRowOfGroup map[int]int, groupSum map[int]float64, lastGroup string, uniqueAGs map[string][]int, greyStyle, whiteStyle *xlsx.Style) error {

	var (
		currentStyle     *xlsx.Style
		value            string
		uniqueValues     map[string]struct{}
		uniqueValueArray []string
		rowIndex, j, k   int
		cell, emptyCell  *xlsx.Cell
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

	setColumnWidths(sheet)

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

func addCell(row *xlsx.Row, value interface{}, style *xlsx.Style) {

	var (
		cell *xlsx.Cell
	)

	cell = row.AddCell()
	cell.SetValue(value)
	cell.SetStyle(style)
}

func addCellWithFormat(row *xlsx.Row, value float64, style *xlsx.Style, format string) {

	var (
		cell *xlsx.Cell
	)

	cell = row.AddCell()
	cell.SetFloatWithFormat(value, format)
	cell.SetStyle(style)
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

func roundFloat(val float64, precision uint) float64 {

	var (
		ratio float64
	)

	ratio = math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
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

func setColumnWidths(sheet *xlsx.Sheet) {

	var (
		widths []float64
		i      int
		width  float64
		newCol *xlsx.Col
	)

	widths = []float64{10, 11, 19.5, 12, 11, 9, 9.5}
	for i, width = range widths {
		newCol = xlsx.NewColForRange(i+1, i+1)
		newCol.SetWidth(width)
		sheet.SetColParameters(newCol)
	}
}
