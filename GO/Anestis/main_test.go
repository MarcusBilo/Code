package main

import (
	"bytes"
	"github.com/LIJUCHACKO/ods2csv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// go test -bench . -benchmem

func BenchmarkFilePath(b *testing.B) {

	var (
		_   string
		err error
		_   []os.FileInfo
	)

	for i := 0; i < b.N; i++ {

		_, err = os.Executable()
		if err != nil {
			handleError(err, "")
			return
		}

	}
}

func BenchmarkReadDir(b *testing.B) {

	var (
		exPath, directory string
		err               error
		_                 []os.FileInfo
	)

	for i := 0; i < b.N; i++ {

		exPath, err = os.Executable()
		if err != nil {
			handleError(err, "")
			return
		}

		directory = filepath.Dir(exPath)

		_, err = readDirectory(directory)
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
		files                       []os.FileInfo
		file                        os.FileInfo
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

		files, err = readDirectory(directory)
		if err != nil {
			handleError(err, directory)
			return
		}

		for _, file = range files {

			buf.Reset() // Reset the buffer before reusing it

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
				return
			}
		}
	}
}

func BenchmarkReadCSV(b *testing.B) {

	var (
		exPath, directory, fileName string
		err                         error
		files                       []os.FileInfo
		file                        os.FileInfo
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

		files, err = readDirectory(directory)
		if err != nil {
			handleError(err, directory)
			return
		}

		for _, file = range files {

			buf.Reset() // Reset the buffer before reusing it

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
		files                                       []os.FileInfo
		file                                        os.FileInfo
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

		files, err = readDirectory(directory)
		if err != nil {
			handleError(err, directory)
			return
		}

		for _, file = range files {

			buf.Reset() // Reset the buffer before reusing it

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
