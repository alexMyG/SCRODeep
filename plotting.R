
data = read.csv("EXECUTION_1549462726.74_REEF_EVOLUTION.csv")


library(dplyr)
library(ggplot2)
data_summar = data %>% 
  group_by(generation) %>%
  summarise_each(funs(mean,sd,se=sd(.)/sqrt(n())))


ggplot(data, aes(x=generation,y=accuracy_validation)) + geom_point(aes(group=generation)) +geom_smooth()


ggplot(data_summar, aes(x=generation)) +
  geom_ribbon(aes(ymin = accuracy_validation_mean -accuracy_validation_sd, ymax = accuracy_validation_mean + accuracy_validation_sd), fill = "grey70") + 
  geom_line(aes(y = accuracy_validation_mean))


df$time = as.numeric(as.character(df$time))

ggplot(df, aes(x = time, y = eat, group = Condition)) +
  geom_smooth(
    aes(fill = Condition, linetype = Condition),
    method = "lm",
    level = 0.65,
    color = "black",
    size = 0.3
  ) +
  geom_point(aes(color = Condition))