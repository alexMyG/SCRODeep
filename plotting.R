library(dplyr)
data = read.csv("EXECUTION_1549462726.74_REEF_EVOLUTION.csv")

data$pos_in_reef <- NULL
data$number_layers <- NULL
data$number_layers2 <- NULL


data <- melt(data, id="generation")
data$variable <- as.character(data$variable)
data <- data %>% dplyr::rowwise() %>% dplyr::mutate(variable = strsplit(variable, split="_")[[1]][2])

data_prepared <- data %>% group_by(generation, variable) %>% summarise_each(funs(mean, sd))


colnames(data_prepared) <- c("Generation", "Data split", "Accuracy", "SD")
ggplot(data_prepared, aes(x=Generation, y=Accuracy, colour=`Data split`)) + 
  geom_line() +
  geom_point()+
  geom_errorbar(aes(ymin=Accuracy-SD, ymax=Accuracy+SD), width=.7,
                position=position_dodge(0.8)) +scale_fill_brewer()








######################################
######################################
######################################

library(data.table)

files  <- list.files(pattern = '\\EXECUTION_.*[0-9].csv')

no_fields = 8

for (no_file in 1:length(files)){
  
  con <- file(files[no_file],open="r")
  lines <- readLines(con)
  close(con)
  
  list_new_lines = list()
  
  for (i in 1:length(lines)){
    
    list_new_lines[[i]] <- strsplit(lines[i],",")[[1]][1:no_fields]
  }
  
  data_new <- plyr::adply(list_new_lines,1,unlist,.id = NULL)
  
  write.table(data_new, gsub(".csv", "_population_evolution.csv", files[no_file]), 
              quote = FALSE, row.names = FALSE, col.names=FALSE, sep=",")
  
}

#list_csvs[[i]] <- data



######################################
######################################
######################################

library(data.table)

files  <- list.files(pattern = '\\EXECUTION_.*population_evolution.csv')

list_csvs <- list()



for (i in 1:length(files)) {
  
  data <- read.csv(files[i])
  
  data$execution = files[i]
  data$generation = seq.int(nrow(data))
  list_csvs[[i]] <- data
}

total_data <- do.call("rbind", list_csvs)

total_data$ratio_reef <- as.character(total_data$ratio_reef)


total_data <- total_data %>% dplyr::rowwise() %>% dplyr::mutate(positions = as.numeric(strsplit(ratio_reef, split="/")[[1]][1]))

total_data$ratio_reef <- NULL

total_data$count_evaluations <- as.numeric(total_data$count_evaluations)
total_data$individuals_depredated <- as.numeric(total_data$individuals_depredated)
total_data$time_generation <- as.numeric(total_data$time_generation)


# total_data_plot <- total_data %>% group_by(generation) %>% summarise_all(funs(mean, sd))

# Evolution fitness depending on execution and generation
ggplot(total_data, aes(x=generation, y=fitness_mean, fill=-generation, group=generation)) + 
  geom_boxplot(alpha=0.2) + theme_minimal() + 
  scale_x_continuous(breaks = round(seq(1, max(total_data$generation), by = 1),1)) +
  scale_y_continuous(breaks = pretty(total_data$fitness_mean, n = 10)) +
  theme(legend.position="none") + xlab("No. generation") + ylab("Population fitness")



# Number of evaluations per generation -> to see how the population evolves
ggplot(total_data, aes(x=generation, y=count_evaluations, fill=-generation, group=generation)) + 
  geom_boxplot(alpha=0.2) + theme_minimal() + 
  scale_x_continuous(breaks = round(seq(1, max(total_data$generation), by = 1),1)) +
  scale_y_continuous(breaks = pretty(total_data$count_evaluations, n = 10)) +
  theme(legend.position="none") + xlab("No. generation") + ylab("Population fitness")


# Time in fitness evaluation -> to see how the time increases
ggplot(total_data, aes(x=generation, y=time_generation, fill=-generation, group=generation)) + 
  geom_boxplot(alpha=0.2) + theme_minimal() + 
  scale_x_continuous(breaks = round(seq(1, max(total_data$generation), by = 1),1)) +
  scale_y_continuous(breaks = pretty(total_data$time_generation, n = 10)) +
  theme(legend.position="none") + xlab("No. generation") + ylab("Population fitness")



# heat map
New executions needed with individuals positions in reef













for (i in 1:length(files)) {

  no_col <- max(count.fields(files[i], sep = ","))
  con <- file(files[i],"r")
  first_line <- readLines(con,n=1)
  close(con)
  cols_file <- strsplit(first_line, ",")[[1]]
  cols_file_total <- append(cols_file, paste0("V",seq_len(no_col-length(cols_file))))
  d1<-read.table(files[i], header = FALSE, sep = ",", col.names = cols_file_total, fill = TRUE)
  
  data <- d1[-1,1:length(cols_file)-1]
  data$execution = files[i]
  data$generation = seq.int(nrow(data))
  list_csvs[[i]] <- data
}


total_data <- ldply(list_csvs, data.frame)

#total_data <- do.call("rbind", list_csvs)

total_data$ratio_reef <- as.character(total_data$ratio_reef)

c <- total_data %>% dplyr::rowwise() %>% dplyr::mutate(positions = strsplit(ratio_reef, split="/")[[1]][1])



#total_data_plot <- do.call(rbind.data.frame, total_data_plot)

total_data_plot$ratio_reef <- NULL
total_data_plot$execution <- NULL

total_data_plot %>% group_by(generation) %>% summarise_if(is.logical,.)


ggplot(total_data_plot, aes(x=generation, y=as.numeric(positions))) + 
 stat_summary(fun.y="mean", geom="line")
  
  
  #geom_errorbar(aes(ymin=Accuracy-SD, ymax=Accuracy+SD), width=.7,
  #              position=position_dodge(0.8)) +scale_fill_brewer()





