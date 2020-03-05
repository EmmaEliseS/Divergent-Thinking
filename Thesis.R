# Required libraries 
library(tidyverse)
library(dplyr)
library(readr)
library(tidytext)
library(nnet)
library(caret)
library(e1071)
library(MLmetrics)
library(irr)
library(ggplot2)

### DATA PREPERATION ###
# Get the files for the first subset of categorized brick data
# file.choose()
setwd("/Users/Emma 1/Documents/Master Thesis/Brick_subset_1")
data_rater1 <- read_csv2("data_subset1_nathalie_klaar.csv")
data_rater2 <- read_csv2("data_subset1_Emma.csv")
data_rater3 <- read_csv2("data_subset1.csv")

data_1 <- left_join(data_rater2, data_rater1, by = c("research_id", "response_id", "respondent_id", "object", 
                                                   "original_response", "cleaned_response")) %>%
  left_join(data_rater3) %>%
  select(-c(category6.x, category7.x, category8.x, category9.x, category10.x, category6.y, category7.y, category8.y,
            category9.y, category10.y, category6, category7, category8, category9, category10))

# Function to find the mode of a vector
Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# Add empty columns to dataframe which need to be filled with Modes of three raters
data_1 <- data_1 %>%
  add_column(label_01 = " ") %>%
  add_column(label_02 = " ") %>%
  add_column(label_03 = " ") %>%
  add_column(label_04 = " ") %>%
  add_column(label_05 = " ")

# Apply function over the rows per column
data_1$label_01 <- apply(data_1[c(7, 12, 17)], 1, Mode)
data_1$label_02 <- apply(data_1[c(8, 13, 18)], 1, Mode)
data_1$label_03 <- apply(data_1[c(9, 14, 19)], 1, Mode)
data_1$label_04 <- apply(data_1[c(10, 15, 20)], 1, Mode)
data_1$label_05 <- apply(data_1[c(11, 16, 21)], 1, Mode)

# Get the files for the second subset of categorized brick data
# file.choose()
setwd("/Users/Emma 1/Documents/Master Thesis/Brick_subset_2")
data_rater4 <- read_csv2("data_subset2_toon.csv")
data_rater5 <- read_csv2("Cat_04_subset2_Viktor (1).csv")
data_rater6 <- read_csv2("data_subset2_fleur.csv")

data_2 <- left_join(data_rater4, data_rater5, by = c("research_id", "response_id", "respondent_id", "object", 
                                                   "original_response", "cleaned_response")) %>%
  left_join(data_rater6) %>%
  select(-c(category6.x, category7.x, category8.x, category9.x, category10.x, category6.y, category7.y, category8.y,
            category9.y, category10.y, category6, category7, category8, category9, category10))

#  Add empty columns to dataframe which need to be filled with Modes of three raters
data_2 <- data_2 %>%
  add_column(label_01 = " ") %>%
  add_column(label_02 = " ") %>%
  add_column(label_03 = " ") %>%
  add_column(label_04 = " ") %>%
  add_column(label_05 = " ")

# Apply function over the rows per column
data_2$label_01 <- apply(data_2[c(7, 12, 17)], 1, Mode)
data_2$label_02 <- apply(data_2[c(8, 13, 18)], 1, Mode)
data_2$label_03 <- apply(data_2[c(9, 14, 19)], 1, Mode)
data_2$label_04 <- apply(data_2[c(10, 15, 20)], 1, Mode)
data_2$label_05 <- apply(data_2[c(11, 16, 21)], 1, Mode)

### FINAL STEP: JOINING THE TWO DATASETS ###
data <- full_join(data_1, data_2) 

# removing duplicate categories
data[is.na(data)] <- " "
data <- data %>%
  filter(label_01 != label_02)

# remove some stuff from environment
rm(data_rater1, data_rater2, data_rater3, data_rater4, data_rater5, data_rater6, data_1, data_2)

########### TEXT MINING: PREPROCESSING WORD2VEC ############

# Drop the na's and select cleaned responses
text <- data %>%
  dplyr::select(cleaned_response) %>%
  drop_na() %>%
  rename(text = cleaned_response)

# Create a tibble 
text_df <- tibble(lines = 1:length(text[[1]]), text = text$text)

# Tokenize the data
tokens <- text_df %>%
  unnest_tokens(word, text)

# Selecting the unique words
tok <- unique(tokens$word)

# Creating a tibble with lines and unique tokens, later used for embeddings
final <- tibble(lines = 1:length(tok), text = tok)

#### WORD2VEC ###
# Pre trained word embeddings
# Load word embeddings from wikipedia dataset
# file.choose()
word2vec_dir <- '/Users/Emma 1/Downloads/'
lines <- readLines(file.path(word2vec_dir, "nlwiki_20180420_300d.txt.bz2"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

# Create embedding layers
embedding_dim <- 300
max_words <-length(tok) + 1
word_index = final$text
i <- 1
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in word_index) {
  if (i < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      # Words not found in the embedding index will be all zeros.
      embedding_matrix[i+1,] <- embedding_vector
  }
  i = i + 1
}

# Transform embedding matrix into dataframe including words
df <- as.data.frame(embedding_matrix) %>%
  slice(-1) %>%
  add_column(text = final$text) %>%
  mutate(sum = rowSums(.[1:300]))

# Selecting the words that have vectors containing only 0's
null <- df %>%
  filter(sum == 0) %>%
  dplyr::select(text)

# Selecting the words containing only filled vectors, so non 0
datafr <- df %>%
  filter(sum != 0) %>%
  dplyr::select(-sum)

##### SENTENCE EMBEDDINGS / DOC2VEC ######
# Sentence embeddings
max_words <- length(text$text)
embedding_sentence <- array(0, c(max_words, embedding_dim))

for(i in 1:length(text$text)){
  sentWords <- filter(tokens, lines == i)
  sentWords <- sentWords$word
  sentAvg <- vector(length = 300)
  for(word in sentWords){
    if(length(which(null == word)) > 0) { # skip words that have vector 0
      sentAvg <- sentAvg + rep.int(900, 300)
    } else {
      st <- filter(df, text == word)
      sentAvg <- as.numeric(st[1:300]) + sentAvg
    }
  }
  sentAvg <- sentAvg / length(sentWords)
  embedding_sentence[i,] <- sentAvg
}

# Create a data frame containing the sentence embeddings and adding column with the sentences
embedding_sentencep <- as.data.frame(embedding_sentence) %>%
  add_column(text = text$text) %>%
  add_column(label_01 = data$label_01) %>%
  add_column(label_02 = data$label_02) %>%
  add_column(label_03 = data$label_03) %>%
  add_column(label_04 = data$label_04) %>%
  add_column(label_05 = data$label_05) %>%
  add_column(rater1.x = data$category1.x) %>%
  add_column(rater11.x = data$category2.x) %>%
  add_column(rater111.x = data$category3.x) %>%
  add_column(rater1111.x = data$category4.x) %>%
  add_column(rater11111.x = data$category5.x) %>%
  add_column(rater2.x = data$category1.y) %>%
  add_column(rater22.x = data$category2.y) %>%
  add_column(rater222.x = data$category3.y) %>%
  add_column(rater2222.x = data$category4.y) %>%
  add_column(rater22222.x = data$category5.y) %>%
  add_column(rater3.x = data$category1) %>%
  add_column(rater33.x = data$category2) %>%
  add_column(rater333.x = data$category3) %>%
  add_column(rater3333.x = data$category4) %>%
  add_column(rater33333.x = data$category5) %>%
  add_column(utf = " ") %>%
  mutate(sum = rowSums(.[1:300]))

# Fill column utf with T or F for utf-8
embedding_sentencep$utf <- validUTF8(embedding_sentencep$text)

# Create data file containing sentences with extreme high sums
nonsencep <- embedding_sentencep %>%
  filter(sum >= 300) %>%
  dplyr::select(text)
utfp <- embedding_sentencep %>%
  filter(utf == F) %>%
  dplyr::select(text)

# Create data file with correct sentence embeddings
emb_sent_corp <- embedding_sentencep %>%
  filter(sum <= 300) %>%
  dplyr::select(-sum)
emb_sent_cor2p <- emb_sent_corp %>%
  filter(utf == T)
# Making sure everything is now UTF-8
valid <- as.data.frame(validUTF8(emb_sent_cor2p$text))

# Selecting unique sentences
emb_sent_cor2p <- distinct(emb_sent_cor2p, text, .keep_all = T)
sentencesp <- emb_sent_cor2p$text

# Create dataframe including only the embedding vectors
emb_sent_corp <- emb_sent_cor2p %>%
  dplyr::select(-c(text, label_01, label_02, 
                   label_03,label_04, label_05, rater1.x, rater11.x, rater111.x, rater1111.x, rater11111.x,
                   rater2.x, rater22.x, rater222.x, rater2222.x, rater22222.x, rater3.x, rater33.x, rater333.x, 
                   rater3333.x, rater33333.x, utf))

# Transform the dataframe to a matrix and add responses as rownames
emb_sent_corp <- as.matrix(emb_sent_corp)
row.names(emb_sent_corp) <- sentencesp



#### NEURAL NETWORK ####

# First selecting the categories and transforming it to a matrix suited for "nnet"
trans <- emb_sent_cor2p[302:306]

# Transforming all columns back to numeric
for(i in 1:ncol(trans)){
  
  trans[,i] <- as.numeric(trans[,i])
  
}
str(trans)

# Creating a new matrix in which 0's and 1's will occur instead of category numbers
trans[is.na(trans)] <- -1
mat <- matrix(0, nrow = nrow(trans), ncol = 64)
for (i in 1:nrow(trans)) {
  vec = trans[i,]
  for (j in vec) {
    if(j >= 0) {
      mat[i, (j + 1)] = 1 - (which(vec == j) - 1) * .05
    }
  }
}

mat <- as.data.frame(apply(mat, 2, function(x) ifelse (abs(x) <= .75, 0, 
                                                          ifelse (abs(x) > .75, 1, x))))

# Create training samples (80%)
xtrain <- emb_sent_cor2p[1:1392, 1:300]
ytrain <- mat[1:1392,]

# Create test samples (20%)
xtest <- emb_sent_cor2p[1392:1741, 1:300]
ytest <- mat[1392:1741,]

# My Neural Network for multiple decay values and size values
# decay = .2 and size (80, 100, 120)
mynet1 = nnet(x = xtrain, y = ytrain, size = 80, maxit = 100, decay = .2, MaxNWts = 100000)
mynet2 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .2, MaxNWts = 100000)
mynet3 = nnet(x = xtrain, y = ytrain, size = 120, maxit = 100, decay = .2, MaxNWts = 100000)
# decay = .4 and size (80, 100, 120)
mynet4 = nnet(x = xtrain, y = ytrain, size = 80, maxit = 100, decay = .4, MaxNWts = 100000)
mynet5 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .4, MaxNWts = 100000)
mynet6 = nnet(x = xtrain, y = ytrain, size = 120, maxit = 100, decay = .4, MaxNWts = 100000)
# decay = .6 and size (80, 100, 120)
mynet7 = nnet(x = xtrain, y = ytrain, size = 80, maxit = 100, decay = .6, MaxNWts = 100000)
mynet8 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .6, MaxNWts = 100000)
mynet9 = nnet(x = xtrain, y = ytrain, size = 120, maxit = 100, decay = .6, MaxNWts = 100000)
# decay = .8 and size (80, 100, 120)
mynet10 = nnet(x = xtrain, y = ytrain, size = 80, maxit = 100, decay = .8, MaxNWts = 100000)
mynet11 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .8, MaxNWts = 100000)
mynet12 = nnet(x = xtrain, y = ytrain, size = 120, maxit = 100, decay = .8, MaxNWts = 100000)
# decay = 1 and size (80, 100, 120)
mynet13 = nnet(x = xtrain, y = ytrain, size = 80, maxit = 100, decay = 1, MaxNWts = 100000)
mynet14 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = 1, MaxNWts = 100000)
mynet15 = nnet(x = xtrain, y = ytrain, size = 120, maxit = 100, decay = 1, MaxNWts = 100000)

# dataframe containing models, sizes, decay and thresholds
models <- paste0("mynet", 1:15)
sizes <- rep(c(80,100,120), 5)
decay <- c(.2,.2,.2,.4,.4,.4,.6,.6,.6,.8,.8,.8,1,1,1)
thresholds <- rep(seq(0.001, .5, 0.002), 15)
##
DF <- cbind(models, sizes, decay) %>%
  as.data.frame() %>%
  slice(rep(1:n(), each = 250)) %>%
  cbind(thresholds)

# Predictions
networks <- list(mynet1, mynet2, mynet3, mynet4, mynet5, mynet6, mynet7, mynet8, mynet9, mynet10, mynet11, mynet12, mynet13, mynet14, mynet15)
predictions <- lapply(networks, function(x) {predict(x, newdata = xtest)})

# Thresholds
threshold <- seq(0.001, .5, 0.002)
tr <- list(list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list())
for (i in seq_along(predictions)) {
  for (j in 1:length(threshold)) {
    tr[[i]][[j]] = ifelse(predictions[[i]] < threshold[j], 0, 1)
  }
}

# Rowsums and Means
sums <- vector()
means <- vector()
for (i in 1:15) {
  for (j in 1:250) {
    sums <- c(sums, mean(rowSums(tr[[i]][[j]] * ytest) / rowSums(ytest)))
    means <- c(means, mean(rowMeans(tr[[i]][[j]] == ytest)))
  }
}

# Add means and sums to DF
DF <- DF %>%
  cbind(sums, means) 
# Add column containing the mean of sums and means
DF <- DF %>%
  rowwise() %>%
  mutate(M = mean(c(sums, means))) %>%
  arrange(desc(M))

DF %>%
  arrange(desc(M)) %>%
  slice(1:10)




##### NEXT STEP #####
# Continuing with a size of 100 and a decay range around .2
mynet16 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .05, MaxNWts = 100000)
mynet17 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .1, MaxNWts = 100000)
mynet18 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .15, MaxNWts = 100000)
mynet19 = mynet2
mynet20 = nnet(x = xtrain, y = ytrain, size = 100, maxit = 100, decay = .25, MaxNWts = 100000)


# dataframe containing models, sizes, decay and thresholds
models <- paste0("mynet", 16:20)
sizes <- c(100,100,100,100,100)
decay <- c(.05, .1, .15, .2, .25)
thresholds <- rep(seq(0.001, .5, 0.002), 5)
##
DF2 <- cbind(models, sizes, decay) %>%
  as.data.frame() %>%
  slice(rep(1:n(), each = 250)) %>%
  cbind(thresholds)

# Predictions
networks <- list(mynet16, mynet17, mynet18, mynet19, mynet20)
predictions <- lapply(networks, function(x) {predict(x, newdata = xtest)})

# Thresholds
threshold <- seq(0.001, .5, 0.002)
tr <- list(list(), list(), list(), list(), list())
for (i in seq_along(predictions)) {
  for (j in 1:length(threshold)) {
    tr[[i]][[j]] = ifelse(predictions[[i]] < threshold[j], 0, 1)
  }
}

# Rowsums and Means
sums <- vector()
means <- vector()
for (i in 1:5) {
  for (j in 1:250) {
    sums <- c(sums, mean(rowSums(tr[[i]][[j]] * ytest) / rowSums(ytest)))
    means <- c(means, mean(rowMeans(tr[[i]][[j]] == ytest)))
  }
}

# Add means and sums to DF
DF2 <- DF2 %>%
  cbind(sums, means) 
# Add column containing the mean of sums and means
DF2 <- DF2 %>%
  rowwise() %>%
  mutate(M = mean(c(sums, means))) %>%
  arrange(desc(M))

DF2 %>%
  arrange(desc(M)) %>%
  slice(1:10)

# Compare size 90 with 100
mynet21 = nnet(x = xtrain, y = ytrain, size = 90, maxit = 100, decay = .2, MaxNWts = 100000)
mynet22 = mynet19
mynet23 = nnet(x = xtrain, y = ytrain, size = 70, maxit = 100, decay = .2, MaxNWts = 100000)

# dataframe containing models, sizes, decay and thresholds
models <- paste0("mynet", 21:23)
sizes <- c(90, 100, 70)
decay <- c(.2, .2, .2)
thresholds <- rep(seq(0.001, .5, 0.002), 3)
##
DF3 <- cbind(models, sizes, decay) %>%
  as.data.frame() %>%
  slice(rep(1:n(), each = 250)) %>%
  cbind(thresholds)

# Predictions
networks <- list(mynet21, mynet22, mynet23)
predictions <- lapply(networks, function(x) {predict(x, newdata = xtest)})

# Thresholds
threshold <- seq(0.001, .5, 0.002)
tr <- list(list(), list(), list())
for (i in seq_along(predictions)) {
  for (j in 1:length(threshold)) {
    tr[[i]][[j]] = ifelse(predictions[[i]] < threshold[j], 0, 1)
  }
}

# Rowsums and Means
sums <- vector()
means <- vector()
for (i in 1:3) {
  for (j in 1:250) {
    sums <- c(sums, mean(rowSums(tr[[i]][[j]] * ytest) / rowSums(ytest)))
    means <- c(means, mean(rowMeans(tr[[i]][[j]] == ytest)))
  }
}

# Add means and sums to DF
DF3 <- DF3 %>%
  cbind(sums, means) 
# Add column containing the mean of sums and means
DF3 <- DF3 %>%
  rowwise() %>%
  mutate(M = mean(c(sums, means))) %>%
  arrange(desc(M))

DF3 %>%
  arrange(desc(M)) %>%
  slice(1:10)


# Plots
# Size paramter versus Prediction Accuracy
ggplot(data = DF, mapping = aes(reorder(x = sizes), y = M)) +
  geom_boxplot(notch = T, fill='#A4A4A4', color="black") +
  theme_classic() +
  labs(x = "Size Parameter", y = "Prediction Accuracy")

# Decay parameter versus Prediction Accuracy
ggplot(data = DF, mapping = aes(x = decay, y = M)) +
  geom_boxplot(notch = T, fill='#A4A4A4', color="black") +
  theme_classic() +
  labs(x = "Decay Parameter", y = "Prediction Accuracy")

# Threshold vs Prediction Accuracy
ggplot(data = DF, mapping = aes(x = thresholds, y = M)) +
  geom_jitter(size = .5) +
  labs(x = "Threshold Parameter", y = "Prediction Accuracy") +
  geom_vline(xintercept = .031, linetype = "dashed", color = "darkred") +
  # scale_fill_manual(name = "Model", labels = seq(1,15,1), values = best) +
  theme_classic()
  
# Descriptive plot
matt <- colSums(mat)
matt <- tibble(category = 1:length(matt), count = matt)

ggplot(matt, mapping = aes(category, count)) +
  geom_bar(stat = "identity") +
  theme_classic() +
  labs(x = "Category (1 - 64)", y = "Count") +
  scale_x_continuous(breaks = c(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65)) +
  scale_y_continuous(breaks = c(0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250))


# COMPARING NN VS. EXPERTS
# First create binary matrixes for each rater

### RATER 1 ###
trans1 <- emb_sent_cor2p[307:311]

# Transforming all columns back to numeric
for(i in 1:ncol(trans1)){
  
  trans1[,i] <- as.numeric(trans1[,i])
  
}
str(trans1)

# Creating a new matrix in which 0's and 1's will occur instead of category numbers
trans1[is.na(trans1)] <- -1
mat1 <- matrix(0, nrow = nrow(trans1), ncol = 64)
for (i in 1:nrow(trans1)) {
  vec = trans1[i,]
  for (j in vec) {
    if(j >= 0) {
      mat1[i, (j + 1)] = 1 - (which(vec == j) - 1) * .05
    }
  }
}

mat1 <- as.data.frame(apply(mat1, 2, function(x) ifelse (abs(x) <= .75, 0, 
                                                       ifelse (abs(x) > .75, 1, x))))


### RATER 2 ###
trans2 <- emb_sent_cor2p[312:316]

# Transforming all columns back to numeric
for(i in 1:ncol(trans2)){
  
  trans2[,i] <- as.numeric(trans2[,i])
  
}
str(trans2)

# Creating a new matrix in which 0's and 1's will occur instead of category numbers
trans2[is.na(trans2)] <- -1
mat2 <- matrix(0, nrow = nrow(trans2), ncol = 64)
for (i in 1:nrow(trans2)) {
  vec = trans2[i,]
  for (j in vec) {
    if(j >= 0) {
      mat2[i, (j + 1)] = 1 - (which(vec == j) - 1) * .05
    }
  }
}

mat2 <- as.data.frame(apply(mat2, 2, function(x) ifelse (abs(x) <= .75, 0, 
                                                       ifelse (abs(x) > .75, 1, x))))


### RATER 3 ###
trans3 <- emb_sent_cor2p[317:321]

# Transforming all columns back to numeric
for(i in 1:ncol(trans3)){
  
  trans3[,i] <- as.numeric(trans3[,i])
  
}
str(trans3)

# Creating a new matrix in which 0's and 1's will occur instead of category numbers
trans3[is.na(trans3)] <- -1
mat3 <- matrix(0, nrow = nrow(trans3), ncol = 64)
for (i in 1:nrow(trans3)) {
  vec = trans3[i,]
  for (j in vec) {
    if(j >= 0) {
      mat3[i, (j + 1)] = 1 - (which(vec == j) - 1) * .05
    }
  }
}

mat3 <- as.data.frame(apply(mat3, 2, function(x) ifelse (abs(x) <= .75, 0, 
                                                       ifelse (abs(x) > .75, 1, x))))

# Now select the same subset as the test data, to compare the network with for each rater
rater_1 <- mat1[1392:1741,]
rater_2 <- mat2[1392:1741,]
rater_3 <- mat3[1392:1741,]

# Predictions of best performing nnet
pred <- predict(mynet2, newdata = xtest) 
pred <- as.data.frame(apply(pred, 2, function(x) ifelse (abs(x) >= 0.031, 1, 
                                                               ifelse (abs(x) < .031, 0, x))))

# Now compare each rater to predicitons of nnet
comp1 <- rowSums(rater_1 == pred)
comp2 <- rowSums(rater_2 == pred)
comp3 <- rowSums(rater_3 == pred)

# Mean of the three rows
m1 <- rowMeans(cbind(comp1, comp2, comp3))

# Now compare raters wihtin each other
comp4 <- rowSums(rater_1 == rater_2)
comp5 <- rowSums(rater_1 == rater_3)
comp6 <- rowSums(rater_2 == rater_3)

# Mean of the three raters
m2 <- rowMeans(cbind(comp4, comp5, comp6))

# Comparing the two mean vectors
t_test <- t.test(m1, m2, paired = T)
a <- data.frame(group = "Expert vs. Model", value = m1)
b <- data.frame(group = "Expert vs. Expert", value = m2)
plot.data <- rbind(a,b)

mean(a$value)
mean(b$value)

# Plot comparison
ggplot(plot.data, mapping = aes(x = group, y = value)) +
  geom_boxplot(notch = T, fill='#A4A4A4', color="black") +
  theme_classic() +
  labs(x = " ", y = "Value")
