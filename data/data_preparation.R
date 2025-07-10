library(tidyverse)
library(magrittr)
library(readxl)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df_exp1 <- read_excel("data_super_Exp1.xlsx")
df_exp2 <- read_excel("data_super_Exp2.xlsx")

df_exp1$exp <- 1
df_exp2$exp <- 2

df_exp1$exp_type <- "witness"
df_exp1$condition_2 <- NA
df_exp1$condition_2 <- ifelse(df_exp1$condition == "congruence witness", "congruent", "incongruent")

df_exp2$exp_type <- NA
df_exp2$exp_type[df_exp2$condition == "Kongruenz Witness" | df_exp2$condition == "Inkongruenz Witness"] <- "witness"
df_exp2$exp_type[df_exp2$condition == "Kongruenz Gambling" | df_exp2$condition == "Inkongruenz Gambling"] <- "gambling"
df_exp2$condition_2 <- NA
df_exp2$condition_2 <- ifelse(df_exp2$condition == "Kongruenz Witness" | df_exp2$condition == "Kongruenz Gambling", "congruent", "incongruent")
df_exp2$subject_id <- df_exp2$subject_id + 1000 

df <- rbind(df_exp1, df_exp2)

df %<>%
  select(
    exp, subject_id, exp_type, condition_2, response_coded, `correct response`,
    correct, number_faces, validity, `response time`
  ) %>% 
  rename(
    id = subject_id,
    condition = condition_2,
    resp = response_coded,
    correct_resp = `correct response`,
    rt = `response time`
  ) %>% 
  mutate(
    id = dense_rank(id),
    id = id - 1,
    correct_resp = case_when(
      correct_resp == "Wahrheit" ~ 1,
      correct_resp == "Lüge" ~ 0,
      correct_resp == "Gewinn" ~ 1,
      correct_resp == "Verlust" ~ 0
    ),
    correct = as.numeric(correct),
    number_faces = as.numeric(number_faces) - 1,
    validity = validity - 0.5,
    rt = rt / 1000
  )

# remove rt outliers
df %<>%
  group_by(id) %>%
  mutate(
    rt = ifelse(rt < 0.2, NA, rt),
    rt = ifelse(rt > 10, NA, rt)
  ) %>% 
  mutate(
    rt = ifelse(log(rt) < boxplot(log(rt))$stats[1], NA, rt),
    rt = ifelse(log(rt) > boxplot(log(rt))$stats[5], NA, rt)
  ) %>% 
  ungroup()

summary <- df %>% 
  group_by(validity) %>% 
  summarise(n_cues = number_faces[1]) %>% 
  mutate(n_cues = n_cues + 1)

write_csv(df, "prepared_data.csv")
