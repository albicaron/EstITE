###########################
# Simulated Guide Example #
###########################

# Libraries
library(tidyverse)
library(BART) # Main package including all the version of BART
library(grf)
library(rlearner)
library(latex2exp)

# Options
N = 150
set.seed(1234)

# Directory
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir)


# 1) Simulate 1-covariate example
x <- rnorm(N, 0, 1)
und = -0.2 + 1.5*x
z <- rbinom(N, 1, pnorm( und ) )

summary(pnorm( und ))
hist( pnorm( und ) )
table(z)
cor(z,x)

# Simulate POs and Y
y0 <- 3 + 0.2*x
y1 <- 5.5 - 0.1*x^2 + sin(1.5*x)

ITE <- y1 - y0

y0_n <- rnorm(N, y0, .25)
y1_n <- rnorm(N, y1, .25)

Y <- y1_n
Y[z==0] = y0_n[z==0]

unobsY = y1_n
unobsY[z==1] = y0_n[z==1]
unobsY[z==0] = y1_n[z==0]


# S-Learner BART demonstration ------------------------------------
SBART <-
  wbart(cbind(x, z), Y, nskip = 1000, ndpost = 2000)

SBART_cnt <- colMeans( predict(SBART, cbind(x, ifelse(z == 1, 0, 1))) )

SBART0 = rep(NA, N); SBART1 = rep(NA, N)
SBART0[z==0] = SBART$yhat.train.mean[z==0]
SBART0[z!=0] = SBART_cnt[z!=0]

SBART1[z==1] = SBART$yhat.train.mean[z==1]
SBART1[z!=1] = SBART_cnt[z!=1]

# T-Learner BART ----------------------------------------
TBART0 <-
  wbart(x[z==0], Y[z==0], nskip = 1000, ndpost = 2000)

TBART1 <-
  wbart(x[z==1], Y[z==1], nskip = 1000, ndpost = 2000)


Y0TBART = rep(NA, N); Y1TBART = rep(NA, N)

Y0TBART[z==0] = TBART0$yhat.train.mean
Y1TBART[z==1] = TBART1$yhat.train.mean

TBART0_fit <- colMeans(predict(TBART0, as.matrix(x)))
TBART1_fit <- colMeans(predict(TBART1, as.matrix(x)))

Y_TBART <- Y1TBART
Y_TBART[is.na(Y_TBART)] = Y0TBART[!is.na(Y0TBART)]



# X-Learner BART -----------------------------------------
# Second step: Imputed Treatment Effects
aux0 = predict(TBART0, newdata = as.matrix(x[z==1]))
aux1 = predict(TBART1, newdata = as.matrix(x[z==0]))

D_0 = rep(NA, N); D_1 = rep(NA, N)

D_1[z==1] <- Y[z==1] - colMeans(aux0)
D_0[z==0] <- colMeans(aux1) - Y[z==0]

# Last step: Tau
tau_1_BART <- wbart(x[z==1], D_1[z==1], nskip = 1000, ndpost = 2000)
tau_0_BART <- wbart(x[z==0], D_0[z==0], nskip = 1000, ndpost = 2000)

tau_1 = colMeans(predict(tau_1_BART, newdata = as.matrix(x)))
tau_0 = colMeans(predict(tau_0_BART, newdata = as.matrix(x)))

# PS score
PS_BART <- pbart(x, z, nskip = 6000, ndpost = 2000)

PS <- PS_BART$prob.train.mean


# Weighting
Tau_final = PS*tau_0 + (1 - PS)*tau_1


# Put everything in df for pretty plotting -----------------------------------
df = data.frame(Y, 
                unobsY, 
                X = x, 
                SBART0, SBART1,
                TBART0_fit, TBART1_fit,
                Group = factor(z, c(0, 1), c("Control", "Treated")))

df_ITE = data.frame(ITE = ITE,
                    D_0, D_1,
                    tau_1, tau_0,
                    Tau_final,
                    noisITE = y1_n - y0_n,
                    X = x)






# Plot POs and ITE

POs <- 
  ggplot(data = df) + geom_point(aes(X, Y, color = Group)) + theme_light() +
  stat_function(fun = function(x) 3 + 0.2*x, col = "coral4", size = 0.7, linetype = "dashed") +
  stat_function(fun = function(x) 5.5 - 0.1*x^2 + sin(1.5*x), col = "royalblue3", size = 0.7, linetype = "dashed") +
  geom_point(aes(X, unobsY, color = Group), color = "grey30", alpha = 0.25) +
  scale_color_manual(values=c("#F8766D", "royalblue1")) +
  scale_y_continuous(breaks = seq(1, 8, 1)) + scale_x_continuous(breaks = seq(-2, 2, 1)) +
  ylab("Response Y") + xlab("Covariate X") +
  theme(text = element_text(size=13.5)) + guides(color = guide_legend(override.aes = list(size = 2)))

POs

ITE <-
  ggplot(data = df_ITE) + geom_line(aes(X, ITE), size = 0.8, linetype = "dashed") + 
  geom_point(aes(X, noisITE), color = "grey50", alpha = 0.4) + theme_light() +
  xlab("Covariate X") + scale_x_continuous(breaks = seq(-2, 2, 1)) +
  scale_y_continuous(breaks = seq(0, 6, 0.5)) +
  theme(text = element_text(size=13.5))

ITE


# Plot model Fit
# SBART
p_SBART <- 
  ggplot(data = df) + geom_point(aes(X, Y, color = Group), alpha = 0.6) + theme_light() +
  scale_color_manual(values=c("#F8766D", "royalblue1")) +
  geom_line(aes(X, SBART0), linetype = "dashed", size = 0.8, color = "grey40") +
  geom_line(aes(X, SBART1), linetype = "dashed", size = 0.8, color = "grey40") +
  scale_y_continuous(breaks = seq(1, 9, 1)) + scale_x_continuous(breaks = seq(-3, 3, 1)) +
  ylab("Response Y") + xlab("Covariate X") +
  theme(legend.position = "none", text = element_text(size=13.5)) + 
  guides(color = guide_legend(override.aes = list(size = 2)))

p_SBART


# TBART
p_TBART <- 
  ggplot(data = df) + geom_point(aes(X, Y, color = Group), alpha = 0.6) + theme_light() +
  scale_color_manual(values=c("#F8766D", "royalblue1")) +
  geom_line(aes(X, TBART0_fit), linetype = "dashed", size = 0.8, color = "coral3") +
  geom_line(aes(X, TBART1_fit), linetype = "dashed", size = 0.8, color = "royalblue3", alpha = 0.7) +
  scale_y_continuous(breaks = seq(1, 9, 1)) + scale_x_continuous(breaks = seq(-3, 3, 1)) +
  ylab("") + xlab("Covariate X") +
  theme(text = element_text(size=13.5)) + guides(color = guide_legend(override.aes = list(size = 2)))

p_TBART


# XBART
XLEARN <-
  ggplot(data = df_ITE) + geom_point(aes(X, noisITE), color = "grey50", alpha = 0.4) + 
  geom_point(aes(X, D_0, color = "Control", fill = "Control"), shape = 24, alpha = 0.5, size = 1.5) +
  geom_point(aes(X, D_1, color = "Treated", fill = "Treated"), shape = 24, alpha = 0.6, size = 1.5) +
  theme_light() + xlab("Covariate X") + scale_x_continuous(breaks = seq(-3, 3.5, 1)) +
  scale_color_manual(name = "Group", values = c("Control" = "brown2", "Treated" = "royalblue1"), 
                     labels = c(expression("D" [0]), expression("D" [1]))) +
  scale_fill_manual(name = "Group", values = c("Control" = "brown2", "Treated" = "royalblue1"),
                    labels = c(expression("D" [0]), expression("D" [1]))) +
  scale_y_continuous(breaks = seq(1, 4, 0.5)) + ylab("ITE") +
  theme(text = element_text(size=13.5)) + guides(color = guide_legend(override.aes = list(size = 2)))

XLEARN



Xfit <-
  ggplot(data = df_ITE) + geom_point(aes(X, noisITE), color = "grey50", alpha = 0.4) + 
  geom_line(aes(X, tau_0, color = "CATE_0"), size = 0.8) +
  geom_line(aes(X, tau_1, color = "CATE_1"), size = 0.8) +
  geom_line(aes(X, Tau_final, color = "CATE"), size = 0.8) +
  scale_color_manual(name = "", values = c("CATE_0" = "brown2", "CATE_1" = "royalblue1", "CATE" = "darkolivegreen4"), 
                     labels = unname(TeX(c("$\\tau (x)", "$\\tau_0 (x)", "$\\tau_1 (x)")))) +
  theme_light() + xlab("Covariate X") + scale_x_continuous(breaks = seq(-3, 3.5, 1)) +
  scale_y_continuous(breaks = seq(1, 4, 0.5)) + ylab("") +
  theme(text = element_text(size=13.5)) + guides(color = guide_legend(override.aes = list(size = 1.5)))

Xfit



# Save them 
ggsave("./Figures/POs.pdf", 
       plot = POs, device = "pdf", width = 14, height = 11, units = "cm")

ggsave("./Figures/ITE.pdf", 
       plot = ITE, device = "pdf", width = 12, height = 11, units = "cm")



ggsave("./Figures/SBART.pdf", 
       plot = p_SBART, device = "pdf", width = 11.5, height = 11, units = "cm")

ggsave("./Figures/TBART.pdf", 
       plot = p_TBART, device = "pdf", width = 14, height = 11, units = "cm")


ggsave("./Figures/XBART1.pdf", 
       plot = XLEARN, device = "pdf", width = 14, height = 11, units = "cm")

ggsave("./Figures/XBART2.pdf", 
       plot = Xfit, device = "pdf", width = 14, height = 11, units = "cm")


