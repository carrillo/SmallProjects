#######################
# author: fernando carrillo 
# Oil prices 
#######################

library(ggplot2); library(e1071); library(TTR); library(reshape)

#detach("package:GENEAread", unload=TRUE)
#detach("package:e1071", unload=TRUE)

setwd("~/workspace/SmallProjects/OilPriceAnalysis/data/")
opec_raw <- read.table("opecData.txt",header=T,sep=",")
opec_raw$Date <- as.Date(opec_raw$Date)

#######################
# Calculate returns from price 
#######################
calc_return <- function(df) {
  df_return <- rep(NA,nrow(df))
  for( i in 2:nrow(df) ) { df_return[i] <- (df[["Value"]][i] - df[["Value"]][i-1])/(df[["Value"]][i-1]) }
  df$Returns <- df_return
  return(df)
}

df <- calc_return(opec_raw)
df <- df[2:nrow(df),]



########################
# Get an overview of the data in frequency domain, retaining some time resolution (Gabor transform). 
# TO-DO: Use frequency information per time window for feature engineering. 
########################
quartz()
a <- stft(df$Returns,win=10)
plot(a)


########################
# Engineer some very simple (and correlated) features. 
# Use tailing mean and variance. 
########################
df$Mean180 <- runMean(df$Returns,n=180)
df$Mean90 <- runMean(df$Returns,n=90)
df$Mean30 <- runMean(df$Returns,n=30)
df$Mean10 <- runMean(df$Returns,n=10)

df$Var180  <- runVar(df$Returns, n=180)
df$Var90  <- runVar(df$Returns, n=90)
df$Var30  <- runVar(df$Returns, n=30)
df$Var10  <- runVar(df$Returns, n=10)

write.table(x=df, file="returnsPredict.csv", sep=',',quote=F,row.names=F)

#######################
# Plot for getting a feeling for the data. 
#######################
quartz()
p  <- ggplot(df, aes(x=Date, y=Value))
p  <- p + geom_line()
p <- p + scale_x_date()
show(p)

quartz()
p  <- ggplot(df, aes(x=Date, y=Returns))
p  <- p + geom_line()
p <- p + scale_x_date()
show(p)

quartz()
p  <- ggplot(df, aes(x=Date, y=Var90))
p  <- p + geom_line()
p <- p + scale_x_date()
show(p)

###################
# Model using a highly regularized model. 
###################



