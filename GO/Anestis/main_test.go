package main

import (
	"bytes"
	"github.com/LIJUCHACKO/ods2csv"
	"os"
	"path/filepath"
	"testing"
)

// go test -bench . -benchmem

func BenchmarkFilePath(b *testing.B) {

	var (
		_   string
		err error
	)

	for i := 0; i < b.N; i++ {

		_, err = os.Executable()
		if err != nil {
			handleError(err, "")
			return
		}

	}
}

func BenchmarkPathDir(b *testing.B) {

	var (
		exPath, _ string
		err       error
		_         []string
	)

	for i := 0; i < b.N; i++ {

		exPath, err = os.Executable()
		if err != nil {
			handleError(err, "")
			return
		}

		_ = filepath.Dir(exPath)

	}
}

func BenchmarkReadDir(b *testing.B) {

	var (
		exPath, directory string
		err               error
		_                 []string
	)

	for i := 0; i < b.N; i++ {

		exPath, err = os.Executable()
		if err != nil {
			handleError(err, "")
			return
		}

		directory = filepath.Dir(exPath)

		_, err = getOdsFilesInDirectory(directory)
		if err != nil {
			handleError(err, directory)
			return
		}
	}
}

func BenchmarkWriteContent(b *testing.B) {

	var (
		exPath, directory, fileName string
		err                         error
		files                       []string
		fileContent                 ods.Odsfile
		outputContent               []string
		buf                         bytes.Buffer
	)

	for i := 0; i < b.N; i++ {

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
				return
			}
		}
	}
}

func BenchmarkReadCSV(b *testing.B) {

	var (
		exPath, directory, fileName string
		err                         error
		files                       []string
		fileContent                 ods.Odsfile
		outputContent               []string
		buf                         bytes.Buffer
		_                           [][]string
	)

	for i := 0; i < b.N; i++ {

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
				return
			}
			_, err = readCSVContent(&buf)
			if err != nil {
				handleError(err, fileName)
				continue
			}
		}
	}
}

func BenchmarkFull(b *testing.B) {

	var (
		exPath, directory, fileName, outputFileName string
		err                                         error
		files                                       []string
		fileContent                                 ods.Odsfile
		outputContent                               []string
		buf                                         bytes.Buffer
		records                                     [][]string
	)

	for i := 0; i < b.N; i++ {

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
				return
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
}
