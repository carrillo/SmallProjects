
# Check out this for spatial analysis: https://www.youtube.com/watch?v=tqcZeje_mWE
# Gif for Edelmann pass: http://gfycat.com/AlertAmpleKingfisher

#########################
# Locate tweets on maps 
#########################

# Dependencies 
require(ggmap); library(plyr); library(grid); library(gridExtra); require(hexbin); require(chron); require(scales); 
require(graphics); require(lubridate); require(rMaps)

# Set workind directory
setwd("/Users/carrillo/workspace/SmallProjects/TwitterAnalysis/")

group1 <- "patriots"
group2 <- "ravens"


#########################
# Read and prepare data 
# 1. Load data
# 2. Format date and subset tweets according to date
#########################
tweets <- read.csv("resources/ravensPatriotsTweets.csv", header=TRUE )
events <- transform(read.csv("resources/ravensPatriotsEvents.csv", header=TRUE ), date=as.POSIXct(date))


# Filter tweets
tweets$date  <- gsub(x=as.character(tweets$date), pattern=" [/+][0-9]{4}", replacement="",perl=TRUE )
tweets$posixDate <- as.POSIXct(tweets$date, format="%a %b %d %H:%M:%S %Y", tz="Europe/London", usetz=TRUE)
tweets$posixDate <- format(tweets$posixDate, tz="America/New_York", usetz=TRUE)

#tweets <- tweets[!is.na( tweets$lon ),]  

startDate <- as.POSIXct("Jan 10 15:30:00 2015",format="%b %d %H:%M:%S %Y",tz="America/New_York")
endDate <- as.POSIXct("Jan 10 21:00:00 2015",format="%b %d %H:%M:%S %Y",tz="America/New_York")
tweets <- tweets[tweets$posixDate >= startDate, ]
tweets <- tweets[tweets$posixDate <= endDate, ]


#########################
# Plot timeseries
#########################
binByMinute <- function(tweets,accuracy=1) {
  ix  <- which(colnames(tweets) == "posixDate")
  ixGroup  <- which(colnames(tweets) == "group")
  dates <- data.frame(year=year(tweets[[ix]]), month=month(tweets[[ix]]), day=day(tweets[[ix]]),
                      hour=hour(tweets[[ix]]), minute=minute(tweets[[ix]]), group=tweets[[ixGroup]])
  dates$minute <- round( dates$minute / accuracy )* accuracy
  dateStr <- function(date) { return(paste(c(date[1],date[2],date[3],date[4],date[5]),collapse="-")) }
  
  dates$string <- apply(dates,MARGIN=1,dateStr)
  
  df <- data.frame(date=as.POSIXlt(NA),tweetCount=as.numeric(NA),group=as.character(NA))
  for( i in unique(tweets[,ixGroup]) ) {
    d <- dates[ dates[,6] == i, ]
    t <- table(d$string)
    df <- rbind(df, data.frame(date=as.POSIXlt(names(t),format="%Y-%m-%d-%H-%M"),tweetCount=as.numeric(t),group=i))
  }
  
  return(na.omit(df))
}

binBySecond <- function(tweets,accuracy=10) {
  ix  <- which(colnames(tweets) == "posixDate")
  ixGroup  <- which(colnames(tweets) == "group")
  dates <- data.frame(year=year(tweets[[ix]]), month=month(tweets[[ix]]), day=day(tweets[[ix]]),
                      hour=hour(tweets[[ix]]), minute=minute(tweets[[ix]]), second=second(tweets[[ix]]), group=tweets[[ixGroup]])
  dates$second <- round( dates$second / accuracy )* accuracy
  dateStr <- function(date) { return(paste(c(date[1],date[2],date[3],date[4],date[5],date[6]),collapse="-")) }
  
  dates$string <- apply(dates,MARGIN=1,dateStr)
  
  df <- data.frame(date=as.POSIXlt(NA),tweetCount=as.numeric(NA),group=as.character(NA))
  for( i in unique(tweets[,ixGroup]) ) {
    d <- dates[ dates[,7] == i, ]
    t <- table(d$string)
    df <- rbind(df, data.frame(date=as.POSIXlt(names(t),format="%Y-%m-%d-%H-%M-%S"),tweetCount=as.numeric(t),group=i))
  }
  
  return(na.omit(df))
}

tweetsPerMinute <- binByMinute(tweets,accuracy=1)
tweetsPerSecond <- binBySecond(tweets,accuracy=10)


decissions <- events[events$class == "decission",]
game_stats <- events[events$class == "game_stats",]
scores <- events[events$class == "score",]
special <- events[events$class == "special",]

quartz()
p <- ggplot(tweetsPerSecond, aes(x=date, y=tweetCount, color=group) ) 
p <- p + geom_line()
#for( i in 1:nrow(decissions)) { p <- p + geom_vline(xintercept = as.integer(decissions[i,1]), size = 0.5, color = "darkorange")}
for( i in 1:nrow(scores)) { p <- p + geom_vline(xintercept = as.integer(scores[i,1]), size = 0.1, color = "red")}
#for( i in 1:nrow(special)) { p <- p + geom_vline(xintercept = as.integer(special[i,1]), size = 0.1, color = "blue")}
#for( i in 1:nrow(game_stats)) { p <- p + geom_vline(xintercept = as.integer(game_stats[i,1]), size = 0.1, color = "black")}
p <- p + scale_x_datetime(breaks = date_breaks('1 hours'), labels = date_format('%H:%M'))
show(p)  

library(waved)
n <- nrow(tweetsPerSecond)
k=(2*pi/T)*[0:n/2-1 -n/2:-1]; 
k <- 2*pi*c((0:(n/2-1)),(-n/2):-1 )

ut <- fft(tweetsPerSecond$tweetCount)
plot(fftshift(k), abs(fftshift(ut)),type='l')

filter=exp(10^-6*(-k^2))
plot(fftshift(k),fftshift(filter),type='l')

utf=filter*ut
plot(fftshift(k), abs(fftshift(ut)),type='l')
points(fftshift(k), abs(fftshift(utf)),type='l',col='red')

u <- fft(ut,inverse=T)
uf <- fft(utf,inverse=T)
plot(abs(fftshift(u)),type='l')
points(abs(fftshift(uf)),type='l',col='red')


#########################################
# Analyze commercial breaks. 
#########################################
traces_around_events <- function(binnedTweets, event_dates, seconds_before_event=0, seconds_after_event=120) {
  
  # Group tweets for all grous 
  binnedTweets <- ddply(binnedTweets,"date", function(x) {sum(x$tweetCount)} )
  
  i <- 1
  df <- data.frame( binnedTweets[binnedTweets$date >= (event_dates[[i]] - seconds_before_event) & binnedTweets$date <= (event_dates[[i]]+seconds_after_event),], event=event_dates[[i]] ) 
  for(i in 2:length(event_dates)) {
    df <- rbind( df, data.frame( binnedTweets[binnedTweets$date >= (event_dates[[i]] - seconds_before_event) & binnedTweets$date <= (event_dates[[i]]+seconds_after_event),], event=event_dates[[i]] )  )
  }
  
  df <- transform(df, date=date, event=as.factor(as.character(event)) ) 
  names(df) <- c("date","count","event")
  
  p <- ggplot(df, aes(x=date, y=count, color=event) ) 
  p <- p + geom_line()
  p <- p + scale_x_datetime()
  show(p)  
}

quartz()
traces_around_events(binnedTweets=tweetsPerSecond,event_dates=commercials[commercials$description == "start",1], 
                     seconds_before_event=120, seconds_after_event=120)

library(leafletR)
tweets_geo <- tweets[!is.na( tweets$lon ),] 
tweets_geo <- transform(tweets_geo,lon=round(tweets_geo$lon,digits=4), lat=round(tweets_geo$lat,digits=4))

dat = ddply(tweets_geo, .(lat, lon, group, posixDate), summarise, count = length(date))
dat <- transform(dat, group=as.character(group))


match_start <- as.POSIXct("Jan 03 20:15:00 2015",format="%b %d %H:%M:%S %Y",tz="America/New_York")
match_end <- as.POSIXct("Jan 03 23:26:00 2015",format="%b %d %H:%M:%S %Y",tz="America/New_York")

ravens_beforeMatch.dat <- toGeoJSON(data=dat[which( dat$posixDate < match_start & dat$group == "ravens" ),], dest=tempdir(), name="ravens_prematch")
ravens_match.dat <- toGeoJSON(data=dat[which( dat$posixDate >= match_start & dat$posixDate <= match_end & dat$group == "ravens" ),], dest=tempdir(), name="ravens_match")
ravens_afterMatch.dat <- toGeoJSON(data=dat[which( dat$posixDate > match_end & dat$group == "ravens" ),], dest=tempdir(), name="ravens_postmatch")

steelers_beforeMatch.dat <- toGeoJSON(data=dat[which( dat$posixDate < match_start & dat$group == "steelers" ),], dest=tempdir(), name="steelers_prematch")
steelers_match.dat <- toGeoJSON(data=dat[which( dat$posixDate >= match_start & dat$posixDate <= match_end & dat$group == "steelers" ),], dest=tempdir(), name="steelers_match")
steelers_afterMatch.dat <- toGeoJSON(data=dat[which( dat$posixDate > match_end & dat$group == "steelers" ),], dest=tempdir(), name="steelers_postmatch")



sty.1 <- styleSingle(col="#311D4E", fill="#311D4E", rad=0.5, fill.alpha=0.25)
sty.2 <- styleSingle(col="#F1C817", fill="#F1C817", rad=0.5, fill.alpha=0.25)


map <- leaflet(data=list(ravens_beforeMatch.dat, ravens_match.dat, ravens_afterMatch.dat,
                         steelers_beforeMatch.dat, steelers_match.dat, steelers_afterMatch.dat), dest=tempdir(),
               style=list(sty.1, sty.1, sty.1, sty.2, sty.2, sty.2), popup=list("*", "Name"),incl.data=TRUE, 
               center=c(39.833,-98.583), zoom=4 )

