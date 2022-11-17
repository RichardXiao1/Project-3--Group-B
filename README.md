# Project-3--Group-B

The purpose of this repo is to give an analysis on six different data channels provided in an online popularity dataset. Links to each channel's analysis is provided.

List of packages used:
library(tidyverse)
library(caret)
library(dplyr)
library(corrplot)
library(psych)
library(ggplot2)
library(purr)
library(rmarkdown)

Links:
[lifestyle analysis](lifestyle.html)

[business analysis](bus.html)

[entertainment analysis](entertainment.html)

[socmed analysis](socmed.html)

[tech analysis](tech.html)

[world analysis](world.html)


Code used to render these documents:

channelIDs <- data.frame("lifestyle", "entertainment", "bus", "socmed", "tech", "world")
output_file <- paste0(channelIDs, ".md")
params<- lapply(channelIDs, FUN = function(x){list(channel = x)})
reports <- tibble(output_file, params)
apply(reports, MARGIN = 1,
FUN = function(x){
render(input = "Project 3 Final.Rmd", output_file = x[[1]], params = x[[2]])
})
