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
          #upper=list(params=c(size = 3)),
          upper=list(continuous='blank'), 
          axisLabels="show", 
          params=c(size=0.5)
  )
}

all <- rbind(class_size, age_father, age_mother, public_private, teacher_morale)
colour <- "OECD_member"

pdf(file="~/Projects/Homepage/content/PISA/performance_correlation.pdf", width=7, height=7, useDingbats=F)
plot_pairs(all, c("reading_Mean","math_Mean","science_Mean","OECD_member"), colour)
dev.off()


###########################
# A: Yes they are highly correlated -> Average reading, math and science. 
# We can use a mean performance measure as a good proxy. 
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

##########################
# Q: Do OECD member states perform better than non-member states? 
##########################
all_oecd <- all[all$OECD_member == TRUE,]
all_non_oecd <- all[all$OECD_member == FALSE,]
ks.test(all_oecd$science_Mean, all_non_oecd$science_Mean)
ks.test(all_oecd$math_Mean, all_non_oecd$math_Mean)
ks.test(all_oecd$reading_Mean, all_non_oecd$reading_Mean)

###########################
# A: OECD member state perform significantly better. Smells like confounding variable e.g. GDP 
###########################

##########################
# Q: How does class room size correlate with performance. OECD member vs. non-member. 
##########################
plot_predictor_response <- function(df) {
  p <- ggplot(df, aes(x=as.numeric(Category), y=mean_Mean) )
  p  <- p + geom_point()
  return(p)
}

pdf(file="~/Projects/Homepage/content/PISA/class_size.pdf", width=7, height=7, useDingbats=F)
plot_pairs(class_size, c("OECD_member","mean_Mean"), "Category")
dev.off()
#plot_predictor_response(class_size)
##########################
# A: Affect of class-room size is more pronounced for OECD member states. Optimal class-room size is surprisingly large (31-35 ). 
##########################


##########################
# Q: How does parental age correlate with performance. OECD member vs. non-member. 
##########################
pdf(file="~/Projects/Homepage/content/PISA/age_mother.pdf", width=7, height=7, useDingbats=F)
plot_pairs(age_mother, c("OECD_member","mean_Mean"), "Category")
dev.off() 

pdf(file="~/Projects/Homepage/content/PISA/age_father.pdf", width=7, height=7, useDingbats=F)
plot_pairs(age_father, c("OECD_member","mean_Mean"), "Category")
dev.off() 
##########################
# Optimal age for parents is > 46 which means ~30 when they become parents (PISA measure 15 year old kids)
##########################

##########################
# Q: How does teacher morale correlate with performance. OECD member vs. non-member. 
##########################
pdf(file="~/Projects/Homepage/content/PISA/teacher_motivation.pdf", width=7, height=7, useDingbats=F)
plot_pairs(teacher_morale, c("OECD_member","mean_Mean"), "Category")
dev.off()
##########################
# Performance decreases monotonically with decreasing teacher morale. 
##########################

##########################
# Q: Private vs. public school 
##########################
pdf(file="~/Projects/Homepage/content/PISA/private_public.pdf", width=7, height=7, useDingbats=F)
plot_pairs(public_private, c("OECD_member","mean_Mean"), "Category")
dev.off()
##########################
# Private outperforms public. 
##########################






