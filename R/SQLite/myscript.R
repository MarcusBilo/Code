# Install and load the RSQLite package if not already installed
if(!require(RSQLite)) {
  install.packages("RSQLite")
  library(RSQLite)
}

# Connect to the SQLite database
con <- dbConnect(SQLite(), dbname = "C:/Users/Marcus/PycharmProjects/pythonProject/my_database.db")

# List tables in the database excluding sqlite_sequence
tables <- dbListTables(con)
tables <- tables[tables != "sqlite_sequence"]
print(tables)

# Query data from a specific table
query_result <- dbGetQuery(con, "SELECT * FROM Animal")
print(query_result)

# Close the database connection when finished
dbDisconnect(con)

# "Ctrl + Shift + S" to run the entire script without echoing