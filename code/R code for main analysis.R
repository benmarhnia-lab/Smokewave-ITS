### Heterogeneity in excess respiratory acute care utilization during wildfire smoke waves in California ###
######################################### R code for key analyses ##########################################
################################################# Yiqun Ma #################################################
#################################### University of California San Diego ####################################
############################################## July 18, 2026 ###############################################

#################################
# 1. R code for the Prophet model
#################################
## Run for each county
## data.train is the county-specific training data for stage 1
## fit Prophet model
prophet_spec <- prophet_reg() %>%
  set_engine("prophet",
             seasonality_yearly  = TRUE,
             seasonality_weekly  = TRUE,
             seasonality_daily   = FALSE,
             prior_scale_changepoints = 0.5,
             prior_scale_seasonality = 10) %>%
  set_mode("regression")

prophet_fit <- prophet_spec %>%
  fit(outcome.rate ~ date + holiday,
      data = data.train)

## get residuals to pass to XGBoost
prophet_residuals <- predict(prophet_fit, data) %>%
  bind_cols(data) %>%
  mutate(prophet_residuals = outcome.rate - .pred) %>%
  rename(pred_prophet = .pred)

#################################
# 2. R code for the XGBoost model
#################################
## Run for each smoke wave event
## data.train is the event-specific training data for stage 2
resamples_kfold <- data.train %>%
  time_series_cv(date_var = date,
                 assess = 7,
                 initial = 28, 
                 slice_limit = 5, 
                 skip = 7, 
                 cumulative = TRUE)
## recipe
xgb_rec <- recipe(prophet_residuals ~ ., data = data.train) %>%
  step_timeseries_signature(date) %>% 
  step_fourier(date, period = 7, K = 2) %>% 
  step_rm(matches("(.iso$)|(.xts$)|(.lbl$)"),
          "date_hour", "date_minute", "date_second",
          "date_hour12", "date_am.pm", "date_month", "date_year") %>%
  step_rm(date) %>%
  step_zv(all_predictors()) %>%
  step_naomit(all_predictors(), all_outcomes(), skip = TRUE) %>%
  step_normalize(matches("(index.num$)|(_week$)|(_wday$)"))

## specify model
xgb_spec <- boost_tree(
  mode           = "regression",
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_engine("xgboost")

## grid for tuning
rec_prep <- xgb_rec %>% prep()
x <- juice(rec_prep) %>% dplyr::select(-prophet_residuals)

xgb_params <- extract_parameter_set_dials(xgb_spec) %>%
  update(
    trees          = trees(c(120L, 400L)),
    mtry           = mtry(c(2L, min(10L, ncol(x)))),
    min_n          = min_n(c(5L, 25L)),
    tree_depth     = tree_depth(c(2L, 5L)),
    learn_rate     = learn_rate(range = c(0.02, 0.20)),
    loss_reduction = loss_reduction(range = c(-6, 1)),
    sample_size    = dials::sample_prop(range = c(0.60, 1.00))
  )

set.seed(2025)
xgb_grid <- grid_space_filling(xgb_params, size = 30)  

## workflow
xgb_wflw <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(xgb_rec)

## tune parameters
ctrl <- control_grid(save_pred = TRUE, parallel_over = "resamples", allow_par = TRUE, verbose = F)

set.seed(2025)
tune_results <- tune_grid(xgb_wflw,
                          resamples = resamples_kfold,
                          grid      = xgb_grid,
                          metrics   = metric_set(rmse, mae),
                          control   = ctrl)
  

best_params <- tune_results %>%
  select_best(metric = "rmse")

## finalize workflow with best params
wflw_xgb_final <- finalize_workflow(xgb_wflw,
                                    best_params)


####################################
# 3. R code for the meta regression
####################################
## var.i is the effect modifier to examine
## excess.rate10k is the estimated excess respiratory acute care encounters per 10,000 people for each smoke wave event

## main analysis
mg <- mvmeta(excess.rate10k ~ var.i + county,
             excess.rate10k.SE,
             data=data, control=list(showiter=F))

## no county fixed effects
mg <- mvmeta(excess.rate10k ~ var.i,
             excess.rate10k.SE,
             data=data, control=list(showiter=F))

## adjust for season
mg <- mvmeta(excess.rate10k ~ var.i + county + season,
             excess.rate10k.SE,
             data=data, control=list(showiter=F))