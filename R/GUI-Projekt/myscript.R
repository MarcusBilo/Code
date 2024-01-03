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

cat("\n")

# Query data from a specific table
query_result <- dbGetQuery(con, "SELECT * FROM Animal")
print(query_result)

cat("\n")

# Query data from a specific table
query_result <- dbGetQuery(con, "SELECT * FROM Location")
print(query_result)

cat("\n")

# Query data from a specific table
query_result <- dbGetQuery(con, "SELECT * FROM Observation")
print(query_result)

cat("\n")









get_count <- function(genus) {
  query <- paste("SELECT COUNT(*) AS Number_of_", genus,"
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  result <- dbGetQuery(con, query)
  return(result)
}


get_average <- function(genus, column) {

  query <- paste("SELECT AVG(Observation.", column, ") AS Average_", genus, "_", column,"
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  result <- dbGetQuery(con, query)
  return(result)
}


get_median <- function(genus, column) {
  
  query <- paste("SELECT MEDIAN(Observation.", column, ") AS Median_", genus, "_", column,"
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  result <- dbGetQuery(con, query)
  return(result)
}

specific_genus <- "deer"
selected_column <- "Age"

count_value <- get_count(specific_genus)
print(count_value)

average_value <- get_average(specific_genus, selected_column)
print(average_value)

median_value <- get_median(specific_genus, selected_column)
print(median_value)









# Close the database connection when finished
dbDisconnect(con)

# "Ctrl + Shift + S" to run the entire script without echoing
