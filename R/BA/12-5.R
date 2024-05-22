# Einfach mit CTRL + ENTER von oben nach unten durchgehen

# We denote electricity consumption as Q
# The montly average of cooling degree days as CLDD_avg
# and the montly average of heating degree days as HTDD_avg
# we work with monthly data
# there are no gaps in the data
# normally it is advised to work off of changes, but given the task
# we work off of the absolute values instead of differences

# setup ------------------------------------------------------------------------
cat("\014")
rm(list=ls())
options(readr.show_col_types = FALSE)
packages <- c("dplyr", "tidyr", "lubridate", "ggplot2", "gridExtra", "readr", "scales", "sandwich", "lmtest")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}
data_in <- paste0(getwd(), "/")
electricity <- read.csv(paste0(data_in,"electricity_resid_AZ.csv"), stringsAsFactors = TRUE)
electricity <- electricity %>% mutate(date = parse_date_time(as.character(MY), orders = "my"))
electricity <- electricity %>% mutate(year = year(as.Date(electricity$date,"%Y/%m/%d", tz="cet")),
                                      month = month(as.Date(electricity$date,"%Y/%m/%d", tz="cet")),
                                      ym = format(electricity$date,'%Ym%m'))
electricity <- electricity %>% dplyr::select(-c("MY","year","month"))
climate <- read_csv(paste0(data_in,"climate_Phoenix_AZ.csv"))
climate <- climate %>% mutate(tempdate = parse_date_time(DATE, orders = "ym"))
climate <- climate %>% mutate(year = year(as.Date(tempdate,"%Y/%m/%d", tz="cet")),
                              month = month(as.Date(tempdate,"%Y/%m/%d", tz="cet")),
                              ym = format(tempdate, '%Ym%m'))

climate <-  climate %>% mutate(ndays = ifelse(month %in% c(1, 3, 5, 7, 8, 10, 12),31, ifelse(month==2,28,30)))
climate <- climate %>% mutate_at(c("CLDD", "HTDD", "DX70", "DX90"),list(avg=~./ndays))
climate <- climate %>% dplyr::select(-c("DATE", "tempdate", "STATION", "NAME"))
data <- inner_join(climate,electricity,by = "ym")
rm(electricity, climate)
data <- data %>% filter(year>=2001 & year <= 2017)
data_out <- paste0(getwd(), "/")
write_csv(data, paste0(data_out, "electricity_AZ_workfile.csv"))

# data exploration -------------------------------------------------------------

data <- read_csv(paste0(data_out, "electricity_AZ_workfile.csv"))
data <- data %>% mutate(date=as.Date(date))
print(data %>% dplyr::select(Q, CLDD_avg, HTDD_avg) %>% summary())

# stromverbrauch ---------------------------------------------------------------
Q_plot <- ggplot(data = data, aes(x = date, y = Q)) +
  geom_line(color = "black", size = 0.7) +
  ylab("Residential electricity consumption (GWh)") +
  xlab("Date") +
  scale_y_continuous(limits = c(1000, 5000), breaks = seq(1000, 5000, 1000)) +  
  scale_x_date(breaks = as.Date(c("2001-01-01", "2003-01-01", "2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01")),
               limits = as.Date(c("2001-01-01", "2017-12-31")), labels = scales::date_format("%b%Y"))
plot(Q_plot)

# rot = es ist heiß - man muss kühlen ------------------------------------------
CLDD_avg_plot <- ggplot(data = data, aes(x = date, y = CLDD_avg)) +
  geom_line(color = "red", size = 0.7) +
  ylab("Degree Days (Fahrenheit)") +
  xlab("Date") +
  scale_y_continuous(expand = c(0.01,0.01), limits = c(0, 34), breaks = seq(0, 34, 2)) +  
  scale_x_date(breaks = as.Date(c("2001-01-01", "2003-01-01", "2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01")),
               limits = as.Date(c("2001-01-01", "2017-12-31")), labels = scales::date_format("%b%Y"))
plot(CLDD_avg_plot)

# blau = es ist kalt - man muss heizen -----------------------------------------
HTDD_avg_plot <- ggplot(data = data, aes(x = date, y = HTDD_avg)) +
  geom_line(color = "blue", size = 0.7) +
  ylab("Degree Days (Fahrenheit)") +
  xlab("Date") +
  scale_y_continuous(expand = c(0.01,0.01), limits = c(0, 34), breaks = seq(0, 34, 2)) +  
  scale_x_date(breaks = as.Date(c("2001-01-01", "2003-01-01", "2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01")),
               limits = as.Date(c("2001-01-01", "2017-12-31")), labels = scales::date_format("%b%Y"))
plot(HTDD_avg_plot)

# ------------------------------------------------------------------------------
combined_CLDD_HTDD_plot <- ggplot(data = data) +
  geom_line(aes(x = date, y = CLDD_avg), color = "red", size = 0.7) +
  geom_line(aes(x = date, y = HTDD_avg), color = "blue", size = 0.7) +
  ylab("Degree Days (Fahrenheit)") +
  xlab("Date") +
  scale_y_continuous(expand = c(0.01,0.01), limits = c(0, 34), breaks = seq(0, 34, 2)) +  
  scale_x_date(breaks = as.Date(c("2001-01-01", "2003-01-01", "2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01")),
               limits = as.Date(c("2001-01-01", "2017-12-31")), labels = scales::date_format("%b%Y"))
grid.arrange(
  Q_plot,
  combined_CLDD_HTDD_plot,
  ncol = 1
)

# ------------------------------------------------------------------------------
Q_2017 <- ggplot(data = data, aes(x = date, y = Q)) +
  geom_line(color = "black", size = 0.7) +
  ylab("Residential electricity consumption (GWh)") +
  xlab("Date") +
  scale_y_continuous(limits = c(1000, 5000), breaks = seq(1000, 5000, 1000)) +  
  scale_x_date(breaks = seq(as.Date("2017-01-01"), as.Date("2017-12-31"), by = "1 month"), 
               limits = as.Date(c("2017-01-01", "2017-12-31")), labels = scales::date_format("%b%Y"))
combined_CLDD_HTDD_plot_2017 <- ggplot(data = data) +
  geom_line(aes(x = date, y = CLDD_avg), color = "red", size = 0.7) +
  geom_line(aes(x = date, y = HTDD_avg), color = "blue", size = 0.7) +
  ylab("Degree Days (Fahrenheit)") +
  xlab("Date") +
  scale_y_continuous(expand = c(0.01,0.01), limits = c(0, 34), breaks = seq(0, 34, 2)) +  
  scale_x_date(breaks = seq(as.Date("2017-01-01"), as.Date("2017-12-31"), by = "1 month"), 
               limits = as.Date(c("2017-01-01", "2017-12-31")), labels = scales::date_format("%b%Y"))
grid.arrange(
  Q_2017,
  combined_CLDD_HTDD_plot_2017,
  ncol = 1
)

# ------------------------------------------------------------------------------
subset_data_5y <- data[data$date >= as.Date("2012-01-01") & data$date <= as.Date("2016-12-31"), ]
model_5y <- lm(Q ~ CLDD_avg + HTDD_avg, data = subset_data_5y)
subset_data_5y$predicted_Q_1 <- predict(model_5y, newdata = subset_data_5y)
plot_5y <- ggplot(data = subset_data_5y, aes(x = date)) +
  geom_line(aes(y = Q), color = "black", size = 0.7) +
  geom_line(aes(y = predicted_Q_1), color = "orange", size = 0.7, linetype = "dashed") +
  ylab("Residential electricity consumption (GWh)") +
  xlab("Date") +
  scale_y_continuous(limits = c(1000, 5000), breaks = seq(1000, 5000, 1000)) +  
  scale_x_date(breaks = seq(as.Date("2012-01-01"), as.Date("2017-01-01"), by = "1 year"),
               limits = c(as.Date("2012-01-01"), as.Date("2017-01-01")), 
               labels = scales::date_format("%b%Y")) +
  theme_minimal()
print(plot_5y)
print(coeftest(model_5y, vcov.=NeweyWest(model_5y, prewhite=FALSE, lag=12, verbose=TRUE)))
# The resulting table can be interpreted as:
# Intercept Estimate + CLDD_avg Estimate * Value + HTDD_avg * Actual Value = Prediction

# ------------------------------------------------------------------------------
model_5y_with_month <- lm(Q ~ CLDD_avg + HTDD_avg + as.factor(month), data = subset_data_5y)
subset_data_5y$predicted_Q_2 <- predict(model_5y_with_month, newdata = subset_data_5y)
plot_5y_with_month <- ggplot(data = subset_data_5y, aes(x = date)) +
  geom_line(aes(y = Q), color = "black", size = 0.7) +
  geom_line(aes(y = predicted_Q_2), color = "orange", size = 0.7, linetype = "dashed") +
  ylab("Residential electricity consumption (GWh)") +
  xlab("Date") +
  scale_y_continuous(limits = c(1000, 5000), breaks = seq(1000, 5000, 1000)) +  
  scale_x_date(breaks = seq(as.Date("2012-01-01"), as.Date("2017-01-01"), by = "1 year"),
               limits = c(as.Date("2012-01-01"), as.Date("2017-01-01")), 
               labels = scales::date_format("%b%Y")) +
  theme_minimal()
plot(plot_5y_with_month)

# ------------------------------------------------------------------------------
grid.arrange(
  plot_5y,
  plot_5y_with_month,
  ncol = 1
)

# ------------------------------------------------------------------------------
model_full <- lm(Q ~ CLDD_avg + HTDD_avg, data = data)
data$predicted_Q_1 <- predict(model_full, newdata = data)
plot_full <- ggplot(data = data, aes(x = date)) +
  geom_line(aes(y = Q), color = "black", size = 0.7) +
  geom_line(aes(y = predicted_Q_1), color = "orange", size = 0.7, linetype = "dashed") +
  ylab("Residential electricity consumption (GWh)") +
  xlab("Date") +
  scale_y_continuous(limits = c(1000, 5000), breaks = seq(1000, 5000, 1000)) +  
  scale_x_date(breaks = as.Date(c("2001-01-01", "2003-01-01", "2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01")),
               limits = as.Date(c("2001-01-01", "2017-12-31")), labels = scales::date_format("%b%Y")) +
  theme_minimal()
model_full_with_month <- lm(Q ~ CLDD_avg + HTDD_avg + as.factor(month), data = data)
data$predicted_Q_2 <- predict(model_full_with_month, newdata = data)
plot_full_with_month <- ggplot(data = data, aes(x = date)) +
  geom_line(aes(y = Q), color = "black", size = 0.7) +
  geom_line(aes(y = predicted_Q_2), color = "orange", size = 0.7, linetype = "dashed") +
  ylab("Residential electricity consumption (GWh)") +
  xlab("Date") +
  scale_y_continuous(limits = c(1000, 5000), breaks = seq(1000, 5000, 1000)) +  
  scale_x_date(breaks = as.Date(c("2001-01-01", "2003-01-01", "2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01")),
               limits = as.Date(c("2001-01-01", "2017-12-31")), labels = scales::date_format("%b%Y")) +
  theme_minimal()
grid.arrange(
  plot_full,
  plot_full_with_month,
  ncol = 1
)

# What confounding variable(s) are we not including that contribute to the increase between 2001 and 2007?
# More installed AC Units? A more lenient use of AC? Some other factor?
# Unclear, however what is clear is the fact that the model only based on Month, Heating and Cooling Days is not capable
# or assessing the increase in the early 2000s; however it is very much capable to asses the current, quite stable,
# level of energy consumption based on Temperature and Month. As such we can potentially assume that the factor
# that lead to an overall increase in energy consumption was only temporary and is no either no longer or only 
# weakly affecting the energy consumption.

#EoF