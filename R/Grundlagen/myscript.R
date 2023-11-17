data <- c(2, 4, 6, 8, 10)

mean_value <- mean(data)
cat("Mean:", mean_value, "\n")

median_value <- median(data)
cat("Median:", median_value, "\n")

sd_value <- sd(data)
cat("Standard Deviation:", sd_value, "\n")

sqrt_result <- sqrt(25)
log_result <- log(2, 10)
sin_result <- sin(pi/2)

cat("√25:", sqrt_result, "\n")
cat("Log2(10):", log_result, "\n")
cat("Sinus(π):", sin_result, "\n")

sum_result <- sum(c(1, 2, 3, 4, 5))
prod_result <- prod(c(1, 2, 3, 4, 5))

cat("Σ(1..5):", sum_result, "\n")
cat("Π(1..5):", prod_result, "\n")
cat("", "\n")

vector_result <- numeric(5)

for (i in 1:5) {
  x <- i
  y <- 3
  
  if (x %% 2 == 0) {
    result <- x + y
    cat("Iteration:", i, "- Sum Result:", result, "\n")
  } else {
    result <- x * y
    cat("Iteration:", i, "- Product Result:", result, "\n")
  }
  
  equal_result <- x == y
  vector_result[i] <- equal_result
}

cat("Vector Result:", vector_result, "\n")

