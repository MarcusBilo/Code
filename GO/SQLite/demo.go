package main

import (
	"database/sql"
	"fmt"
	_ "modernc.org/sqlite"
)

type User struct {
	ID   int
	Name string
	Age  int
}

func closeDB(db *sql.DB) {
	err := db.Close()
	if err != nil {
		fmt.Println("Error closing database:", err)
	}
}

func closeRows(rows *sql.Rows) {
	err := rows.Close()
	if err != nil {
		fmt.Println("Error closing rows:", err)
	}
}

func printUsers(rows *sql.Rows) {
	fmt.Println("\nUsers:")
	for rows.Next() {
		var user User
		if err := rows.Scan(&user.ID, &user.Name, &user.Age); err != nil {
			fmt.Println("Error scanning row:", err)
			return
		}
		fmt.Printf("ID: %d, Name: %s, Age: %d\n", user.ID, user.Name, user.Age)
	}
	if err := rows.Err(); err != nil {
		fmt.Println("Error iterating rows:", err)
	}
}

func main() {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		fmt.Println("Error opening database:", err)
		return
	}
	defer closeDB(db)

	_, err = db.Exec("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, age INTEGER);")
	if err != nil {
		fmt.Println("Error creating table:", err)
		return
	}
	_, err = db.Exec("INSERT INTO users (name, age) VALUES (\"Alice\", 30), (\"Bob\", 25), (\"Charlie\", 35);")
	if err != nil {
		fmt.Println("Error inserting users:", err)
		return
	}
	rows, err := db.Query("SELECT id, name, age FROM users")
	if err != nil {
		fmt.Println("Error querying database:", err)
		return
	}
	defer closeRows(rows)

	printUsers(rows)
}
