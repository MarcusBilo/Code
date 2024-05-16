package main

import (
	"bytes"
	"encoding/csv"
	ods "github.com/LIJUCHACKO/ods2csv"
	"github.com/tealeg/xlsx/v3"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func main() {
	executablePath, err := os.Executable()
	if err != nil {
		os.Exit(1)
	}
	directory := filepath.Dir(executablePath)
	dir, err := os.Open(directory)
	if err != nil {
		os.Exit(1)
	}
	defer func(dir *os.File) {
		err := dir.Close()
		if err != nil {
		}
	}(dir)

	files, err := dir.Readdir(-1)
	if err != nil {
		os.Exit(1)
	}

	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".ods") {
			fileContent, err := ods.ReadODSFile(directory + "/" + file.Name())
			if err != nil {
				continue
			}
			var buffer bytes.Buffer

			for _, sheet := range fileContent.Sheets {
				for _, row := range sheet.Rows {
					for i, cell := range row.Cells {
						buffer.WriteString(cell.Text)
						if i < len(row.Cells)-1 {
							buffer.WriteString(";")
						}
					}
					buffer.WriteString("\n")
				}
			}

			reader := csv.NewReader(&buffer)
			reader.Comma = ';'
			records, err := reader.ReadAll()
			if err != nil {
				return
			}

			uniqueEntries := findUniqueEntries(records)

			if err := saveAsXlsx(file.Name(), uniqueEntries, records); err != nil {
				return
			}
		}
	}
}

func findUniqueEntries(records [][]string) map[string][]int {
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

func saveAsXlsx(fileName string, uniqueEntries map[string][]int, records [][]string) error {
	wb := xlsx.NewFile()
	sheet, err := wb.AddSheet("Entries")
	if err != nil {
		return err
	}

	row := sheet.AddRow()
	row.AddCell().Value = "Artikelnummer"
	row.AddCell().Value = "Artikelbezeichnung"
	row.AddCell().Value = "Anzahl Bestellungen"
	row.AddCell().Value = "Durchschnittliche Menge je Bestellung"
	row.AddCell().Value = "Durchschnittlicher Bestellpreis"
	row.AddCell().Value = "Gesamtkosten"

	for entry, rows := range uniqueEntries {
		row := sheet.AddRow()
		row.AddCell().Value = entry
		row.AddCell().Value = records[rows[0]][10]
		row.AddCell().SetInt(len(rows))

		sumAmount := 0
		sumPrice := 0
		for _, row := range rows {
			amount, _ := strconv.Atoi(records[row][12])
			sumAmount += amount
			price, _ := strconv.Atoi(records[row][11])
			sumPrice += price
		}
		row.AddCell().SetInt(sumAmount / len(rows))
		row.AddCell().SetInt(sumPrice / len(rows))
		row.AddCell().SetInt((sumAmount / len(rows)) * (sumPrice / len(rows)) * len(rows))
	}

	base := fileName[:len(fileName)-len(filepath.Ext(fileName))]
	if err := wb.Save(base + ".xlsx"); err != nil {
		return err
	}
	return nil
}
