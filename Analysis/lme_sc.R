rm(list = ls(all.names = TRUE))
library(nlme)
library(emmeans)

# Parameters
base_path = "F:\\MNIOpen\\Results\\sc_broadband\\"
data_name = "sc_all_stages"
save_name = "sc_bb_"
stages = c("W", "N3", "R")
distances = seq(10, 90, by=10)  # c(10, 30, 50, 70, 90)

# Read data
df = read.csv(paste0(base_path, data_name, ".csv"))
# Select stages
df = df[df$stage %in% stages,]
# Add 'pair' column
df$pair = paste0(df$ch_1, "-", df$ch_2)
# Add factors
df$pat <- factor(df$pat)
df$stage <- factor(df$stage, levels=stages)
df$pair <- factor(df$pair)


###
# 1) Fit overall profiles
###
df_lme = df[, c("pat", "stage", "pair", "corr")]
LME <- lme(corr ~ stage,
           random = ~ 1 | pat/pair, data = df_lme,
           control = lmeControl(opt = "optim")
)
# Save in txt
sink(paste0(base_path, save_name, "all.txt"))
print(summary(LME))
sink()  # returns output to the console


###
# 2) Fit per region
###
Regions = unique(df$region_1)
W_vals = rep(NA, length(Regions))
N3_vals = rep(NA, length(Regions))
R_vals = rep(NA, length(Regions))
for (i in 1:length(Regions)){
  reg = Regions[i]
  print(reg)
  df_reg = df[(df$region_1 == reg) | (df$region_2 == reg), ]
  df_reg = df_reg[, c("pat", "stage", "pair", "corr")]
  LME <- lme(corr ~ 0 + stage,
             random = ~ 1 | pat/pair, data = df_reg,
             control = lmeControl(opt = "optim")
  )
  Coefs = summary(LME)$coefficients$fixed
  W_vals[i] = Coefs["stageW"]
  N3_vals[i] = Coefs["stageN3"]
  R_vals[i] = Coefs["stageR"]
}
df_res = data.frame(region = Regions,
                    W = W_vals,
                    N3 = N3_vals,
                    R = R_vals)
write.csv(df_res, paste0(base_path, save_name, "regions.csv"), row.names=FALSE)


###
# 3) Fit per region and across distances
###
df_list = list() # rep(NA, length(distances))
for (j in 1:length(distances)){
  
  df_dist = df[(df$dist >= distances[j] - 10) & (df$dist < distances[j] + 10), ]
  
  Regions = unique(df_dist$region_1)
  W_vals = rep(NA, length(Regions))
  N3_vals = rep(NA, length(Regions))
  R_vals = rep(NA, length(Regions))
  for (i in 1:length(Regions)){
    reg = Regions[i]
    df_reg = df_dist[(df_dist$region_1 == reg) | (df_dist$region_2 == reg), ]
    df_reg = df_reg[, c("pat", "stage", "pair", "corr")]
    
    
    LME <- lme(corr ~ 0 + stage,
               random = ~ 1 | pat/pair, data = df_reg,
               control = lmeControl(opt = "optim")
    )
    Coefs = summary(LME)$coefficients$fixed
    W_vals[i] = Coefs["stageW"]
    N3_vals[i] = Coefs["stageN3"]
    R_vals[i] = Coefs["stageR"]
  df_res = data.frame(
    distance = distances[j],
    region = Regions,
    W = W_vals,
    N3 = N3_vals,
    R = R_vals)
  df_list[[j]] = df_res
  
  }
}
df_distances = do.call("rbind", df_list)
write.csv(df_distances, paste0(base_path, save_name, "dists.csv"), row.names=FALSE)
