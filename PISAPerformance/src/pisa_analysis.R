#####################
# author: fernando carrillo 
# PISA data-set 
#####################

library( "ggplot2" ); require("GGally")

setwd("~/workspace/SmallProjects/PISAPerformance/data/")

add_OECD_member_feature <- function(df) {
  index_first_non_OECD <- which(df$Country == "Albania")[1]
  df$OECD_member <- as.factor(c(rep(TRUE,index_first_non_OECD-1), rep(FALSE,(nrow(df)-index_first_non_OECD+1))))  
  return(df)
}

class_size <- read.table("class_sizeClean.CSV",sep=",",head=T,na.strings=c("a  ","m"))
class_size$Category <- as.factor(class_size$Category)
class_size <- add_OECD_member_feature(class_size)

age_father <- read.table("parental_age_father_clean.CSV",sep=",",head=T,na.strings=c("a  ","m"))
age_father$Category <- as.factor(age_father$Category)
age_father <- add_OECD_member_feature(age_father)

age_mother <- read.table("parental_age_mother_clean.CSV",sep=",",head=T,na.strings=c("a  ","m"))
age_mother$Category <- as.factor(age_mother$Category)
age_mother <- add_OECD_member_feature(age_mother)

public_private <- read.table("public_vs_private2_clean.CSV",sep=",",head=T,na.strings=c("a  ","m"))
public_private$Category <- as.factor(public_private$Category)
public_private <- add_OECD_member_feature(public_private)

teacher_morale <- read.table("teacher_morale_clean.CSV",sep=",",head=T,na.strings=c("a  ","m"))
teacher_morale$Category <- as.factor(teacher_morale$Category)
teacher_morale <- add_OECD_member_feature(teacher_morale)


###########################
# Q1: Are the target values correlated? 
###########################
plot_pairs <- function(df, columnNames, colour) {
  ggpairs(data=df, 
          columns=which(names(df) %in% columnNames ), 
          colour=colour, 
          lower=list(continuous='points'), 
          upper=list(continuous='blank'), 
          axisLabels="none", 
          params=c(size=0.5)
  )    
}

all <- rbind(class_size, age_father, age_mother, public_private, teacher_morale)
colour <- "OECD_member"
plot_pairs(all, c("all_Mean","reading_Mean","math_Mean","science_Mean","OECD_member"), colour)

###########################
# A: Yes they are highly correlated -> Average reading, math and science. 
# OECD member states have a higher average performance in all topics. TO-DO: Test for significance: Kolmogorov-Smirnov
###########################
summarize_subjects  <- function(df) {
  df$mean_Mean <- apply(df, MARGIN=1, FUN=function(x) {
    mean( as.numeric(c(x[["reading_Mean"]], x[["math_Mean"]], x[["science_Mean"]])), na.rm=T )
  } )
  columnCount <- ncol(df)
  df <- df[,c(1:(columnCount-2),columnCount,columnCount-1)]
  return(df)
}

class_size <- summarize_subjects(class_size)
age_father <- summarize_subjects(age_father)
age_mother <- summarize_subjects(age_mother)
public_private <- summarize_subjects(public_private)
teacher_morale <- summarize_subjects(teacher_morale)

quartz()
all <- rbind(class_size, age_father, age_mother, public_private, teacher_morale)
plot_pairs(all, c("all_Mean","reading_Mean","math_Mean","science_Mean","OECD_member","mean_Mean"), "OECD_member")

##########################
# Q: How does class room size correlate with performance. OECD member vs. non-member. 
##########################
plot_predictor_response <- function(df) {
  p <- ggplot(df, aes(x=as.numeric(Category), y=mean_Mean) )
  p  <- p + geom_point()
  return(p)
}
plot_pairs(class_size, c("all_Mean","reading_Mean","math_Mean","science_Mean","OECD_member","mean_Mean"), "Category")
#plot_predictor_response(class_size)
##########################
# A: Affect of class-room size is more pronounced for OECD member states. Optimal class-room size is surprisingly large (31-35 ). 
#     No obvious difference for math, science, reading. Flatter profiles for non-OECD member states. 
##########################


##########################
# Q: How does parental age correlate with performance. OECD member vs. non-member. 
##########################
plot_pairs(age_mother, c("all_Mean","reading_Mean","math_Mean","science_Mean","OECD_member"), "Category")
plot_pairs(age_father, c("all_Mean","reading_Mean","math_Mean","science_Mean","OECD_member"), "Category")
##########################
# Optimal age mother 51 and older (non_OECD) and 46 – 50 years (OECD). Optimal age father 51 and older (non_OECD) and 46 – 50 years (OECD). 
# Similar profile for reading, math and science. 
##########################


##########################
# Q: How does teacher morale correlate with performance. OECD member vs. non-member. 
##########################
plot_pairs(teacher_morale, c("all_Mean","reading_Mean","math_Mean","science_Mean","OECD_member"), "Category")
##########################
# Performance decreases monotonically with decreasing teacher morale. 
##########################






