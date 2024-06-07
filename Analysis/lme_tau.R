rm(list = ls(all.names = TRUE))
library(nlme)
library(emmeans)

# Parameters
base_path = "F:\\MNIOpen\\Results\\timescales_gamma\\"
data_name = "tau_stages"
save_name = "timescales_gamma_"
stages = c("W", "N3", "R")

# Read data
df = read.csv(paste0(base_path, data_name, ".csv"))
# Select stages
df = df[df$stage %in% stages,]
# Add factors
df$pat <- factor(df$pat)
df$stage <- factor(df$stage, levels=stages)
df$chan <- factor(df$chan)


###
# 1) Fit overall profiles
###
df_lme = df[, c("pat", "stage", "chan", "region", "tau")]
LME <- lme(tau ~ stage,
           random = ~ 1 | pat/chan, data = df_lme,
           control = lmeControl(opt = "optim")
)
# Save in txt
sink(paste0(base_path, save_name, "all.txt"))
print(summary(LME))
sink()  # returns output to the console


###
# 2) Fit per region
###
Regions = unique(df$region)
W_vals = rep(NA, length(Regions))
N3_vals = rep(NA, length(Regions))
R_vals = rep(NA, length(Regions))
for (i in 1:length(Regions)){
  reg = Regions[i]
  print(reg)
  df_reg = df[df$region == reg, ]
  df_reg = df_reg[, c("pat", "stage", "chan", "tau")]
  LME <- lme(tau ~ 0 + stage,
             random = ~ 1 | pat/chan, data = df_reg,
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

