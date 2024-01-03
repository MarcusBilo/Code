# Install and load the RSQLite package
if(!require(RSQLite)) {
  install.packages("RSQLite")
  library(RSQLite)
}

# Install and load the ggplot2 package
if(!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}

# Connect to the database
con <- dbConnect(SQLite(), dbname = "C:/Users/Marcus/PycharmProjects/pythonProject/my_database.db")

query_result <- dbGetQuery(con, "SELECT * FROM Animal")
print(query_result)

cat("\n")

query_result <- dbGetQuery(con, "SELECT * FROM Location")
print(query_result)

cat("\n")

query_result <- dbGetQuery(con, "SELECT * FROM Observation")
print(query_result)

cat("\n")









get_count <- function(genus) {
  query <- paste("SELECT Observation.ID
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  data <- dbGetQuery(con, query)
  count <- length(data$ID)
  return(paste("Number of", genus, "observations:", count))
}


get_count_all_animals <- function() {
  get_count <- function(genus) {
    query <- paste("SELECT Observation.ID
                    FROM Observation 
                    JOIN Animal ON Observation.AnimalID = Animal.ID 
                    WHERE Animal.Genus = ", shQuote(genus), sep = "")
    
    data <- dbGetQuery(con, query)
    count <- length(data$ID)
    return(count)
  }
  
  unique_genera <- dbGetQuery(con, "SELECT DISTINCT Genus FROM Animal")$Genus
  counts <- list()
  
  for (genus in unique_genera) {
    count <- get_count(genus)
    counts[[genus]] <- count
  }
  
  return(counts)
}


get_average <- function(genus, column) {
  query <- paste("SELECT Observation.", column, "
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  data <- dbGetQuery(con, query)
  value <- mean(data[[column]])
  return(paste("Average for", column, "of", genus, "is", value))
}


get_median <- function(genus, column) {
  query <- paste("SELECT Observation.", column, "
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  data <- dbGetQuery(con, query)
  value <- median(data[[column]])
  return(paste("Median for", column, "of", genus, "is", value))
}


get_stdev <- function(genus, column) {
  query <- paste("SELECT Observation.", column, "
                  FROM Observation 
                  JOIN Animal ON Observation.AnimalID = Animal.ID 
                  WHERE Animal.Genus = ", shQuote(genus), sep = "")
  
  data <- dbGetQuery(con, query)
  stdev <- sd(data[[column]])
  return(paste("Stdev for", column, "of", genus, "is", stdev))
}

specific_genus <- "deer"
selected_column <- "Age"

count_value <- get_count(specific_genus)
print(count_value)
cat("\n")

average_value <- get_average(specific_genus, selected_column)
print(average_value)
cat("\n")

median_value <- get_median(specific_genus, selected_column)
print(median_value)
cat("\n")

stdev_value <- get_stdev(specific_genus, selected_column)
print(stdev_value)
cat("\n")



# bar charts using ggplot for:
# 1) Die jeweiligen Alter eines Genus darstellen -> function: get all ages of specific genus




all_animal_counts <- get_count_all_animals()

animal_counts_df <- data.frame(Genus = names(all_animal_counts), Count = unlist(all_animal_counts))

plot <- ggplot(animal_counts_df, aes(x = Genus, y = Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = NULL, x = "Genus", y = "Count") +
  theme(axis.text.x = element_text(vjust = 0.5, hjust=0.5))

ggsave("animal_counts_plot.png", plot)





# Close the database connection when finished
dbDisconnect(con)

# "Ctrl + Shift + S" to run the entire script without echoing
