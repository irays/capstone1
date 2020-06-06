# File Name: HarvardX_Capstone1_June04_2020.R
# Title: "Movielens Capstone Project HarvardX"
# Author: "MFR"
# Date : June 04, 2020
#================================================================================#
# Preparing R Workspace environment by deleting files filled with data and values, 
# which you can all delete using the following line of code:
# rm(list=ls())
#================================================================================#
# A call to gc causes a garbage collection to take place.
# invisible(gc())
#================================================================================#  
# Loading the required libraries
#================================================================================#  
library(dslabs)
library(dplyr)
library(tidyverse)
library(lubridate)
library(stringr)
library(rvest)
library(tidytext)
library(wordcloud)
library(doParallel)
library(recosystem)
library(RColorBrewer)
library(plotly)
library(ggthemes)
library(data.table)
library(kableExtra)
library(Matrix.utils)
library(DT)
library(irlba)
library(recommenderlab)
library(tidyverse)
library(lubridate)
library(dplyr)
library(knitr)
library(kableExtra)
library(matrixStats)
library(reticulate)
#=====================================================================#
# Downloading MovieLens 10M dataset and create edx set, validation set
#=====================================================================#
# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Data Exploration
# 1. Similarity Measures Using Cosine Similarity usersId and movieId converted as factors 
# for analysis. To perform this transformation we make a copy of edx set, since we want 
# to keep unchanged our original training set.(edx)

edx2 <- edx

edx2$userId <- as.factor(edx2$userId)
edx2$movieId <- as.factor(edx2$movieId)

# The output is a sparse matrix of class dgcMatrix.
# Data needs to be converted using userId & movieId into numeric vectors.

edx2$userId <- as.numeric(edx2$userId)
edx2$movieId <- as.numeric(edx2$movieId)

sparse_ratings <- sparseMatrix(i = edx2$userId,
                               j = edx2$movieId ,
                               x = edx2$rating, 
                               dims = c(length(unique(edx2$userId)),
                                        length(unique(edx2$movieId))),  
                      dimnames = list(paste("u", 1:length(unique(edx2$userId)), sep = ""), 
                      paste("m", 1:length(unique(edx2$movieId)), sep = "")))


# Remove the edx2 created
rm(edx2)

# Take a look on the first 10 users
sparse_ratings[1:10,1:10]

# Convert rating matrix into a recommenderlab sparse matrix
ratingMat <- new("realRatingMatrix", data = sparse_ratings)
ratingMat

# Compute user similarity using the cosine similarity

similarity_users <- similarity(ratingMat[1:50,], 
                               method = "cosine", 
                               which = "users")

image(as.matrix(similarity_users), main = "User similarity")

# Computing similarity between  movies using the cosine similarity.

similarity_movies <- similarity(ratingMat[,1:50], 
                                method = "cosine", 
                                which = "items")

image(as.matrix(similarity_movies), main = "Movies similarity")
#=======================================================================# 
# 2. Dimension Reduction
# Dimensionality reduction techniques such as "pca" and "svd" are useful. 
# Transforming the original high-dimensional space into a lower-dimension.
# Dimension reduction using Irlba package, which it is a fast and memory-efficient 
# way to compute a partial SVD. The augmented implicitly 
# restarted Lanczos bidiagonalization algorithm (IRLBA) finds a few approximate 
# largest (or, optionally, smallest) singular values and 
# corresponding singular vectors of a sparse or dense matrix using a method of 
# Baglama and Reichel.

set.seed(1)
Y <- irlba(sparse_ratings,tol=1e-4,verbose=TRUE,nv = 100, maxit = 1000)

# plot singular values

plot(Y$d, pch=20, col = "green", cex = 1.5, xlab='Singular Value', ylab='Magnitude', 
     main = "Singular Values for User-Movie Matrix")

# calculate sum of squares of all singular values
all_sing_sq <- sum(Y$d^2)

# variability described by first 6, 12, and 20 singular values
first_six <- sum(Y$d[1:6]^2)
print(first_six/all_sing_sq)

perc_vec <- NULL
for (i in 1:length(Y$d)) {
  perc_vec[i] <- sum(Y$d[1:i]^2) / all_sing_sq
}

plot(perc_vec, pch=20, col = "blue", cex = 1.5, xlab='Singular Value', ylab='% of Sum of Squares of Singular Values', main = "Choosing k for Dimensionality Reduction")
lines(x = c(0,100), y = c(.90, .90))

# To find the exact value of k, i calculate  the length of the vector that remains 
# from our running sum of squares after excluding any items within that vector that exceed 0.90.
k = length(perc_vec[perc_vec <= .90])
k
# Decomposition of Y ; matrices U, D, and V accordingly:

U_k <- Y$u[, 1:k]
dim(U_k)

D_k <- Diagonal(x = Y$d[1:k])
dim(D_k)

V_k <- t(Y$v)[1:k, ]
dim(V_k)

#=======================================================================# 
# A Linear model with Average rating and different BIASES are used as 
# Predictors in this Algorithm 
#=======================================================================# 
# Y_hat = mu + b_i + b_u + b_g + b_t 
# mu = Average rating of all movies 
# b_i = Bias based on Movies 
# b_u = Bias based on Users 
# b_g = Bias based on Genres 
# b_t = Bias based on Date the movie is rated 
#=======================================================================#

# A function that computes the RMSE for vectors of ratings and their corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
# Average of all ratings
mu<-mean(edx$rating)

# Compute the RMSE for the model and store it in a Dataframe	
naive_rmse<-RMSE(validation$rating,mu)		
RMSE_Results<-data_frame(method="Average RMSE", RMSE=naive_rmse)	

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Histogram of movie bias
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
#=======================================================================#	
# Improve the predicted value of mu with movies effect	
#=======================================================================#
pred_movie_effect <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% .$b_i		
pred_movie_effect <- pred_movie_effect + mu		
#===========================================================================#	
# Compute the RMSE with movies effect model	
#===========================================================================#		
movie_effect<-RMSE(validation$rating,pred_movie_effect)		

# Creating a results table with this naive approach:
RMSE_Results<-bind_rows(RMSE_Results,data_frame(method="Movie Effect Model",RMSE=movie_effect))		

# Histogram of user bias
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

pred_user_effect <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs,by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% .$pred		

user_effect <-RMSE(validation$rating,pred_user_effect)	

# Compute the RMSE with movie + user effect	
#=========================================================================#	
	
RMSE_Results<-bind_rows(RMSE_Results,data_frame(method="Movie + User Effects Model",RMSE=user_effect))		

# Considering genres effect on RMSE
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 30000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Error bar plots by genres" , caption = "Genre analysis on edx dataset") 

genre_avgs <-edx %>% 
  left_join(movie_avgs,by="movieId") %>% 
  left_join(user_avgs,by="userId")%>%group_by(genres) %>%	  
  summarize(b_g = mean(rating - mu - b_i - b_u))

pred_genre_effect <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 	  
  left_join(user_avgs,by='userId') %>% 
  left_join(genre_avgs, by='genres') %>% 
  mutate(pred_genre = mu + b_i + b_u + b_g) %>% .$pred_genre

genre_effect <-RMSE(validation$rating,pred_genre_effect)	

# Compute the RMSE with movie + user effect	+ genres
RMSE_Results<-bind_rows(RMSE_Results,data_frame(method="Movie + User + Genres Effect",RMSE=genre_effect))	
#=======================================================================================================#
# Compute the RMSE for the model with Year effect and store it in a Dataframe
# Creating date column using as_datetime function in the lubridate package.
edx <- mutate(edx, date = as_datetime(timestamp))

# Extracting the year rated
edx <- mutate(edx, year_rated = year(as_datetime(timestamp)))
head(edx)

edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, week")+
  labs(subtitle = "Average Ratings",
       caption = "Data Analysis on edx dataset")

#Extracting the title year
title_year <- stringi::stri_extract(edx$title, regex = "(\\d{4})", comments = TRUE ) %>% as.numeric()

#Add the title year
edx_with_title_year <- edx %>% mutate(title_year = title_year)
head(edx_with_title_year)

#=======================================================================================#
# Drop the timestamp
edx_with_title_year <- edx_with_title_year %>% select(-timestamp)

head(edx_with_title_year)
#=======================================================================================#
validation <- mutate(validation, date = as_datetime(timestamp))

# Extracting the year rated
validation <- mutate(validation, year_rated = year(as_datetime(timestamp)))
head(validation)

#Extracting the title year
title_year <- stringi::stri_extract(validation$title, regex = "(\\d{4})", comments = TRUE ) %>% as.numeric()

#Add the title year
validation_with_title_year <- validation %>% mutate(title_year = title_year)
head(validation_with_title_year)

#=======================================================================================#
# Drop the timestamp
validation_with_title_year <- validation_with_title_year %>% select(-timestamp)

head(validation_with_title_year)
#=======================================================================================#
edx_with_title_year %>% group_by(title_year) %>%	  
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%	  
  filter(n >= 10000) %>% 	mutate(title_year = reorder(title_year, avg)) %>%	  
  ggplot(aes(x = title_year, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 	  
  geom_point() + geom_errorbar() + theme(axis.text.x = element_text(angle = 90, hjust = 1))	
#=======================================================================================#
year_avgs<-edx_with_title_year %>% 
  left_join(movie_avgs,by="movieId") %>% 
  left_join(user_avgs,by="userId") %>% left_join(genre_avgs,by="genres") %>%	
  group_by(title_year) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))

pred_year_effect <- validation_with_title_year %>% 
  left_join(movie_avgs, by='movieId') %>% 	  
  left_join(user_avgs,by='userId') %>% 
  left_join(genre_avgs, by='genres') %>% 
  left_join(year_avgs,by="title_year") %>%
  mutate(pred_year = mu + b_i + b_u + b_g) %>% .$pred_year

year_effect <-RMSE(validation_with_title_year$rating,pred_year_effect)	

# Compute the RMSE with movie + user effect	+ genres + year effects

RMSE_Results<-bind_rows(RMSE_Results,data_frame(method="Movie + User + Genres + Year Effect",RMSE=year_effect))

#=======================================================================================================#
# Regularization
# Here are the 10 largest mistakes:
validation_with_title_year  %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) 

movie_titles <- edx_with_title_year %>% 
  select(movieId, title) %>%
  distinct()

# Here are the 10 best movies according to our estimate:
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) 

# And here are the 10 worst:
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) 

#================================================================================================#
# Penalized Least Squares
# Let's compute these regularized estimates of b_i using lambda = 3.Later, we will see why we picked 3.
lambda <- 3
mu <- mean(edx_with_title_year$rating)
movie_reg_avgs <- edx_with_title_year %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# To see how the estimates shrink, let's make a plot of the regularized estimates versus 
# the least squares estimates.
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Now, let's look at the top 10 best movies based on the penalized estimates  

predicted_ratings <- validation_with_title_year %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>% .$pred

user_reg_effect <- RMSE(validation_with_title_year$rating,predicted_ratings)
RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = user_reg_effect))
#========================================================================================#

lambdas <- seq(0, 10, 0.25)

mu <- mean(edx_with_title_year$rating)
just_the_sum <- edx_with_title_year %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation_with_title_year %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>% .$pred
  return(RMSE(validation_with_title_year$rating,predicted_ratings))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_with_title_year$rating)
  
  b_i <- edx_with_title_year %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_with_title_year %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation_with_title_year %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% .$pred
  
  return(RMSE(validation_with_title_year$rating,predicted_ratings))
})

qplot(lambdas, rmses) 

lambda <- lambdas[which.min(rmses)]
lambda

RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
#====================================================================================#
kable(RMSE_Results, format = "rst", row.names = FALSE)
#====================================================================================#
RMSE_Results %>% mutate(name = reorder(method, desc(RMSE))) %>% 
  ggplot(aes(x = method, y = RMSE, fill = RMSE)) +
  geom_bar(stat = "identity") + coord_flip() + scale_fill_distiller(palette = "PiYG") + 
  ggtitle("RMSE RESULTS")

#====================================================================================#
# Matrix Factorization with Stochastic Gradient Descent (SGD)
#====================================================================================#
# A call to gc causes a garbage collection to take place.
invisible(gc())
#====================================================================================#
edx_new <- edx_with_title_year %>% select(-c("genres","title","date","year_rated","title_year"))

edx_new <- edx_with_title_year %>% select(movieId, userId, rating)
validation_new <- validation_with_title_year %>% select(-c("genres","title","date","year_rated","title_year"))

validation_new <- validation_with_title_year %>% select(movieId, userId, rating)

edx_matrix <- as.matrix(edx_new)
dim(edx_matrix)
validation_matrix <- as.matrix(validation_new)
dim(validation_matrix)
write.table(edx_matrix, file = "trainingset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE)

write.table(validation_matrix, file = "validationset.txt", sep = " ", 
            row.names = FALSE, col.names = FALSE)

#  data_file(): Specifies a data set from a file in the hard disk. 

set.seed(1)
training_dataset <- data_file("trainingset.txt")

validation_dataset <- data_file("validationset.txt")

# The usage of 'recosystem' package is really straightforward. Here the training_dataset 
# variable represents a dataframe with userid, movieid, rating columns. 
# Parameters are set arbitrarily: the number of factors (dim) is 30, regularization for P 
# and Q factors (costp_l2, costq_l2) is set to 0.001, and convergence can be controlled by a number of 
# iterations (niter = 10) and learning rate (lrate = 0.1). The user can also control the parallelization using 
# the nthread = 6 parameter:

recommender = Reco()

# Matrix Factorization :  tuning training set

opts <- recommender$tune(training_dataset, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                     costp_l1 = 0, costq_l1 = 0,
                                     nthread = 1, niter = 10))

# Model training 
recommender$train(training_dataset, opts = c(opts$min , nthread = 1, niter = 50,verbose=FALSE))

pred_file = tempfile()

recommender$predict(validation_dataset, out_file(pred_file))

real_ratings <- read.table("validationset.txt", header = FALSE, sep = " ")$V3

# Mean squared error (abbreviated MSE) and root mean square error (RMSE) refer to the 
# amount by which the values predicted by an estimator differ from the quantities 
# being estimated (typically outside the sample from which the model was estimated).

# We calculate the standard deviation of the residuals (prediction errors) RMSE . 
# Between the predicted ratings and the real ratings . If one or more predictors 
# are significant, the second step is to assess how well the model fits the data 
# by inspecting the Residuals Standard Error (RSE).

pred_ratings <- scan(pred_file)
dim(as.matrix(pred_ratings))

mf_pred <- data.frame(real_ratings[1:20], pred_ratings[1:20])
kable(head(mf_pred,20), format = "pandoc", caption = "Top twenty rows of edx dataset")
#======================================================================================#
RMSE_MF <- RMSE(real_ratings, pred_ratings)

RMSE_Results <- bind_rows(RMSE_Results,
                          data_frame(method="MF with SGD",  
                                     RMSE = RMSE_MF))

kable(RMSE_Results, format = "rst", row.names = FALSE)
RMSE_Results %>% mutate(name = reorder(method, desc(RMSE))) %>% 
  ggplot(aes(x = method, y = RMSE, fill = RMSE)) +
  geom_bar(stat = "identity") + coord_flip() + scale_fill_distiller(palette = "PiYG") + 
  ggtitle("RMSE RESULTS")
#=======================================================================================#