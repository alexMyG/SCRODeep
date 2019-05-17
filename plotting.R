library(plyr)
library(dplyr)
library(RColorBrewer)
library(data.table)
library(ggplot2)






####################################
######################################
######################################

# Removing individuals encoding from files, generating new ones ending in "_population_evolution.csv"
files  <- list.files(pattern = '\\EXECUTION_.*[0-9].csv')

no_fields = 16

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
# General plots

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

count_executions_generation <- total_data %>%
  group_by(generation) %>%
  summarise(count=n())


## PLOT 1 
# Evolution fitness boxplot depending on execution and generation in test set
ggplot(total_data) + 
  geom_boxplot(aes(x=generation, y=fitness_mean_test, fill=-generation, group=generation), lwd=0.2, outlier.size = 0.5,fill="lightblue") +
  theme_minimal() + 
  scale_x_continuous(breaks = round(seq(0, max(total_data$generation), by = 2),1), expand = c(0.01, 0.01)) +
  scale_y_continuous(breaks = pretty(total_data$fitness_mean_test, n = 10), expand = c(0.01, 0.01)) +
  theme(legend.position="none", axis.title=element_text(size=10)) +
  xlab("\nNo. generation") + ylab("Population mean fitness (test)\n") 
ggsave("1_fitness_mean_test_evolution_through_generations_boxplot.pdf", width = 4, height = 4)



## PLOT 2
# All evolution fitness lines fitness mean in test set
ggplot(total_data, aes(x=generation, y=fitness_mean_test, colour=execution)) + 
  geom_line(size=0.1) +
  geom_point(size=0.5) +
  theme_minimal() +
  scale_x_continuous(breaks = round(seq(0, max(total_data$generation), by = 2),1), expand = c(0.01, 0.01)) +
  scale_y_continuous(breaks = pretty(total_data$fitness_mean_test, n = 10), expand = c(0.01, 0.01)) +
  theme(legend.position="none", axis.title=element_text(size=10)) +
  xlab("\nNo. generation") + ylab("Population mean fitness (test)\n") 
ggsave("2_fitness_mean_test_evolution_through_generations_lines.pdf", width = 4, height = 4)

# + geom_text(data = count_executions_generation, aes(label = count, y=1))

# stat_summary(aes(x = generation, y=fitness_mean_train), fun.y=mean, colour="red", geom="line") + 
#   stat_summary(aes(x = generation, y=fitness_mean_train), fun.y=mean, colour="red", geom="point", size=0.5) + 
#   stat_summary(aes(x = generation, y=fitness_mean_test), fun.y=mean, colour="blue", geom="line") + 
#   stat_summary(aes(x = generation, y=fitness_mean_test), fun.y=mean, colour="blue", geom="point", size=0.5) + 
#   stat_summary(aes(x = generation, y=fitness_mean_validation), fun.y=mean, colour="green", geom="line") + 
#   stat_summary(aes(x = generation, y=fitness_mean_validation), fun.y=mean, colour="green", geom="point", size=0.5) + 
#   



total_data_mean_set <- total_data %>% 
  group_by(generation) %>% 
  summarise(mean_validation_test = mean(fitness_mean_test),
            mean_validation_train = mean(fitness_mean_train),
            mean_validation_validation = mean(fitness_mean_validation))


## PLOT 3
# Evolution fitness depending on execution and generation by data split
ggplot(total_data_mean_set, aes(x=generation)) + 
  geom_line(aes(y = mean_validation_test, colour = "Test", linetype="Test")) + 
  geom_line(aes(y = mean_validation_validation, colour = "Validation", linetype="Validation")) + 
  geom_line(aes(y = mean_validation_train, colour = "Training", linetype="Training")) + 
  geom_point(aes(y = mean_validation_train, colour = "Training", shape="Training"), size=0.8) + 
  
  geom_point(aes(y = mean_validation_test, colour = "Test", shape = "Test"), size=1.5) + 
  geom_point(aes(y = mean_validation_validation, colour = "Validation", shape="Validation"), size=0.8) + 
  theme_minimal() +
  scale_x_continuous(breaks = round(seq(0, max(total_data_mean_set$generation), by = 2),1), expand = c(0.01, 0.01)) +
  scale_y_continuous(breaks = pretty(total_data_mean_set$mean_validation_test, n = 10), expand = c(0.01, 0.01)) +
  labs(color  = "Data split", linetype = "Data split", shape = "Data split") +
  xlab("\nNo. generation") + ylab("Population fitness\n") +
  theme(legend.position="bottom", axis.title=element_text(size=10)) +
  scale_colour_brewer(palette="Set2")
ggsave("3_fitness_mean_evolution_through_generations_line_splits.pdf", width = 4, height = 4)




# Plot 4
# Number of new evaluations per generation -> to see how the population evolves
ggplot(total_data, aes(x=generation, y=count_evaluations, group=generation)) + 
  geom_boxplot(fill="lightblue",lwd=0.2, outlier.size = 0.5) + theme_minimal() + 
  scale_x_continuous(breaks = round(seq(0, max(total_data$generation), by = 2),1), expand = c(0.01, 0.01)) +
  scale_y_continuous(breaks = pretty(total_data$count_evaluations, n = 10), expand = c(0.01, 0.01)) +
  theme(legend.position="none", axis.title=element_text(size=10))+
  xlab("\nNo. generation") + ylab("Count new individuals evaluated\n")
ggsave("4_evolution_number_new_individuals.pdf", width =4, height = 4)




# Time in fitness evaluation -> to see how the time increases

total_data$time_generation <- total_data$time_generation/60

# Plot 5
# Time used in each generation 
ggplot(total_data, aes(x=generation, y=time_generation, group=generation)) + 
  geom_boxplot(fill="lightblue", lwd=0.2, outlier.size = 0.5) + theme_minimal() + 
  scale_x_continuous(breaks = round(seq(0, max(total_data$generation), by = 2),1), expand = c(0.01, 0.01)) +
  scale_y_continuous(breaks = pretty(total_data$time_generation, n = 10), expand = c(0.01, 0.01)) +
  theme(legend.position="none", axis.title=element_text(size=10))+
  xlab("\nNo. generation") + ylab("New population evaluation time (minutes)\n")
ggsave("5_fitness_evaluation_time_evolution.pdf", width =4, height = 4)






########################################
########################################
########################################
# BEST INDIVIDUAL OF EACH EXECUTION

files  <- list.files(pattern = '\\EXECUTION_.*REEF_EVOLUTION.csv')


list_csvs <- list()

for (i in 1:length(files)) {
  
  data <- read.csv(files[i])
  
  data$execution = files[i]
  list_csvs[[i]] <- data
}

total_data <- do.call("rbind", list_csvs)


# obtain the last reef for each execution
result <- total_data %>% 
  group_by(execution) %>%
  filter(generation == max(generation)) %>%
  arrange(generation, pos_in_reef, accuracy_validation, number_layers, accuracy_training, accuracy_test, execution)


result_best_individual_by_execution <- result %>% 
  group_by(execution) %>% 
  filter(accuracy_validation == max(accuracy_validation)) %>% 
  arrange(pos_in_reef, accuracy_validation, number_layers, accuracy_training, accuracy_test)



best_individual_validation <- result_best_individual_by_execution[result_best_individual_by_execution$accuracy_validation == max(result_best_individual_by_execution$accuracy_validation),]

best_execution_file_validation <- best_individual_validation$execution
  
result_best_individual_by_execution$execution <- NULL

result_best_individual_by_execution <- as.data.frame(result_best_individual_by_execution)
result_best_individual_by_execution %>% summarise_all(funs(mean))

write.csv(result_best_individual_by_execution %>% summarise_all(.funs = c(mean="mean", sd="sd", min="min", max="max")), "summary_results.csv")







########################################
########################################
########################################

best_execution_file_validation
# PLOT 6: evolution accuracy in three data splits in the best individual

# Fitness evolution in best execution
data = read.csv(gsub("_population_evolution", "_REEF_EVOLUTION", best_execution_file_validation))
data$pos_in_reef <- NULL
data$number_layers <- NULL
data$number_layers2 <- NULL
data <- melt(data, id="generation")
data$variable <- as.character(data$variable)
data <- data %>% dplyr::rowwise() %>% dplyr::mutate(variable = strsplit(variable, split="_")[[1]][2])
data_prepared <- data %>% group_by(generation, variable) %>% summarise_each(funs(mean, sd))
colnames(data_prepared) <- c("Generation", "Data split", "Accuracy", "SD")
data_prepared$`Data split` <- gsub("test", "Test",data_prepared$`Data split`  )
data_prepared$`Data split` <- gsub("training", "Training",data_prepared$`Data split`  )
data_prepared$`Data split` <- gsub("validation", "Validation",data_prepared$`Data split`  )
data_prepared$id <- nrow(data_prepared)
data_prepared_df <- as.data.frame(data_prepared)

ggplot(data_prepared_df) + 
  geom_line(aes(x=Generation, y=Accuracy, colour=`Data split`)) +
  geom_point(aes(x=Generation, y=Accuracy, colour=`Data split`)) +
  geom_errorbar(aes(x=Generation, ymin=Accuracy-SD, ymax=Accuracy+SD, colour=`Data split`), width=.7,
                position=position_dodge(0.8))  + theme_minimal() +
  scale_x_continuous(breaks = round(seq(0, max(data_prepared$Generation), by = 1),1)) +
  scale_y_continuous(breaks = round(seq(0, 1, by = 0.1),1)) +
  scale_color_manual(name = "Data split", values = c("deepskyblue2", "peachpuff4", "orange")) +
  theme(legend.position="bottom") + xlab("\nGeneration") + ylab("Accuracy\n")

ggsave("6_best_individual_fitness_evolution.pdf", width = 8, height = 3.5)





# heat map
#New executions needed with individuals positions in reef

data = read.csv(gsub("_population_evolution", "_REEF_EVOLUTION", best_execution_file_validation))

library(viridis)


colnames(data)<- c("Generation", "Position in reef", "Accuracy\n validation", "No. layers", "Accuracy\n training", "Accuracy\n test")
# PLOT 7
# HEATMAP
ggplot(data, aes(`Position in reef`, Generation)) + 
  geom_tile(aes(fill = `Accuracy\n test`), colour = "white") + 
  geom_text(aes(label=round(`Accuracy\n test`,4)*100), size=2.1) +
  scale_fill_gradient2(name = "Accuracy in\ntest", midpoint = 0.8, low = "khaki1", mid = "white", trans="logit", high = "steelblue4", breaks=c(0.1,0.5,0.85, 0.95, 0.98))+
  theme_minimal() +              #, limits=c(0.1, 1)) + theme_minimal() +
  scale_y_continuous(breaks = round(seq(0, max(data$Generation), by = 1),1), expand=c(0,0),) +
  scale_x_continuous(breaks = round(seq(0, max(data$`Position in reef`), by = 1),1), expand=c(0,0)) +
  theme(panel.background=element_rect(fill="grey22", colour="white")) +
  xlab("Position in reef") + ylab("No. generation") +
  theme(legend.position="none", axis.title=element_text(size=10))
  
  #theme(legend.position="bottom", legend.key.width = unit(1.5,"cm")) +
  
  #theme(axis.text.x = element_text(angle = 90, hjust = 1))
  ggsave("7_evolution_fitness_reef_generations.pdf", width = 10, height = 6)

  
  


  
  