###################
# NHANES Analysis #
###################

rm(list = ls())


### LIBRARIES
library(tidyverse)
library(SparseBCF) # OOB Bayesian Causal Forests
library(BART) # Main package including all the version of BART
library(nnet)

### EVALUATION FUNCTIONS
BScore <- function(x,y) mean((x-y)^2)

set.seed(1234)

# Load data
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./..')

NHANES <- read.csv("./Data/NHANES.csv")

covs.cont.n <- c("age", "RefAge")
covs.cat.n <- colnames(NHANES)[!colnames(NHANES) %in% covs.cont.n][-c(1, 2)]


# Save number of predictors and obs
P <- length(c(covs.cont.n, covs.cat.n))
N  <-  nrow(NHANES)  # save num of obs


# Get data in arrays format
myY <- NHANES$BMI
myZ <- NHANES$School_meal
myX <- as.matrix(NHANES[,c(covs.cont.n,covs.cat.n)])
dim(myX)

# standardize the continuous variables
# myX[, covs.cont.n] <- scale(myX[, covs.cont.n])


# PS estimation ------------------------------------
## PScore Model - 1 hidden layer neural net
PS_nn <- nnet(x = myX, y = myZ, size = 20, maxit = 2000, 
              decay = 0.01, trace=FALSE, abstol = 1.0e-8) 
PS_est = PS_nn$fitted.values

MLmetrics::AUC(PS_est, myZ)


# BCF ----------------------------------------
######################### Normal BCF
#### Train
bcf <-
  SparseBCF(y = myY, z = myZ, x_control = myX, 
            pihat = PS_est, 
            OOB = F, 
            sparse = F,
            update_interval = 5000, 
            nburn = 5000, nsim = 5000)


tau_BCF <- colMeans(bcf$tau)
apply(bcf$tau, 2, var)
summary(tau_BCF)

# # Upload NSGP predictions for pretty plotting -----------------------------------
# tau_NSGP <- read.csv("./Results/GP_tau.csv")[, 1]
# 
# # Correlation
# cor(tau_BCF, tau_NSGP)
# cor(tau_BCF, tau_NSGP, method = "spearman")
# 
# sd(tau_BCF)
# sd(tau_NSGP)


# Plotting ------------------------------------
########### Joyplot
# Get the 10 centiles
p = quantile(PS_est, probs = seq(0, 1, 0.1))
index = c(NA)

for (i in 1:length(p)) {
  index[i] = which(abs(PS_est - p[i]) == (min(abs(PS_est - p[i]))))
}


# melt data
BCF_est = as.data.frame(bcf$tau[, index])
colnames(BCF_est) = names(p)

BCF_est = reshape2::melt(BCF_est)
colnames(BCF_est) = c("PS_cent", "BCF")
df = data.frame(BCF_est)

df_final = reshape2::melt(df)
colnames(df_final) = c("PS_cent", "Model", "CATE")

# Joy plots
library(ggridges)

CATEplot <-
  ggplot(df_final, aes(y = PS_cent, x = CATE, fill = 0.5 - abs(0.5 - stat(ecdf)))) +
  stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE, scale = 1.5, size = 0.75, rel_min_height = 0.05) +
  scale_fill_distiller(palette = "Greens", direction = 1) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_x_continuous(limits = c(-0.8, 0.8), breaks = seq(-0.8, 0.8, 0.2)) + xlab("CATE") +
  theme_ridges(center_axis_labels = TRUE) + ylab("Propensity Percentile") +
  theme(legend.position = "none", text = element_text(size=13.5))

CATEplot



AGE <-
  ggplot(data = data.frame(age = NHANES$age, tau_BCF)) +
  geom_point(aes(x = age, y = tau_BCF, color = "BCF"), alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", size = 0.6) +
  scale_color_manual(name = "", labels = c(expression(paste(hat(tau), "(x"[i], ")"))), 
                     values = c("BCF" = "chartreuse4")) +
  theme_light() + scale_x_continuous(breaks = seq(4, 18, 2)) +
  scale_y_continuous(breaks = seq(-0.1, 0.2, 0.1), limits = c(-0.15, 0.2)) +
  ylab("CATE") + xlab("Child's Age") + theme(text = element_text(size=13.5)) +
  guides(color = guide_legend(override.aes = list(alpha = 0.7, size = 2.5)))


AGE

# AGE <-
#   ggplot(data = data.frame(age = NHANES$age, tau_BCF, tau_NSGP)) +
#   geom_point(aes(x = age, y = tau_BCF, color = "BCF"), alpha = 0.3) +
#   geom_point(aes(x = age, y = tau_NSGP, color = "NSGP"), alpha = 0.3) +
#   geom_hline(yintercept = 0, linetype = "dashed", size = 0.6) +
#   scale_color_manual(name = "Model", values = c("BCF" = "coral2", "NSGP" = "chartreuse4")) +
#   theme_light() + scale_x_continuous(breaks = seq(4, 18, 2)) +
#   scale_y_continuous(breaks = seq(-0.1, 0.3, 0.1), limits = c(-0.15, 0.3)) +
#   ylab("CATE") + xlab("Child's Age") + theme(text = element_text(size=13.5)) +
#   guides(color = guide_legend(override.aes = list(alpha = 0.7, size = 2.5)))
# 
# 
# AGE



# Plot tree
# Change wanted labels
library(rpart)
library(rpart.plot)

colnames(myX)[which(colnames(myX) %in% c("age"))] = 
  c("Age")

mytree <- rpart(
  tau_BCF ~ ., 
  data =  cbind.data.frame(tau_BCF, myX), 
  control = list(maxdepth = 2)
)

pdf("./Results/DecTree.pdf", 
    width = 12, height = 8)
rpart.plot(mytree, type = 2, extra = 101, clip.right.labs = FALSE, 
           box.palette = "Greens", # color scheme
           branch.lty = 3, # dotted branch lines
           shadow.col = "gray",
           branch.lwd = 2,
           tweak = 1.1,
           branch = 1, under = TRUE,  yesno = 2)
dev.off()

summary(myY)


# Save them 
ggsave("./Results/DENS_TAU.pdf", 
       plot = CATEplot, device = "pdf", width = 14, height = 11, units = "cm")

ggsave("./Results/AGE_TAU.pdf", 
       plot = AGE, device = "pdf", width = 14, height = 11, units = "cm")


# Chan (2016) analysis
fit1<- ATE::ATE(myY,myZ,myX)
fit1
summary(fit1)
plot(fit1)
