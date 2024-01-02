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








specific_animal <- "deer"

query <- "SELECT COUNT(*) AS Number_of_Animals
          FROM Observation 
          JOIN Animal ON Observation.AnimalID = Animal.ID 
          WHERE Animal.Genus = ?"

result <- dbGetQuery(con, query, params = list(specific_animal))
print(result)


query <- "SELECT AVG(Observation.Age) AS Average_Age
          FROM Observation 
          JOIN Animal ON Observation.AnimalID = Animal.ID 
          WHERE Animal.Genus = ?"

result <- dbGetQuery(con, query, params = list(specific_animal))
print(result)







# Close the database connection when finished
dbDisconnect(con)

# "Ctrl + Shift + S" to run the entire script without echoing