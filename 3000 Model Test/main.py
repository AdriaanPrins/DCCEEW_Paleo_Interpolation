import os
# 🛑 GAG THE C-LIBRARIES: Force 1 thread per worker to stop CPU hijacking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Google imports
from google.cloud import bigquery

# Kriging Imports
from pykrige.rk import RegressionKriging
from pykrige.ok import OrdinaryKriging

# SKlearn and Model imports
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Other Models
from pyinterpolate import inverse_distance_weighting
from pygam import LinearGAM, s, l, te

# Standard Tools Imports
import gc
import csv
import pandas as pd
import numpy as np
from datetime import date, timedelta
import concurrent.futures
import multiprocessing


# ==================================================
# GLOBAL DICTIONARIES & HELPER FUNCTIONS
# ==================================================
Month_dict = {
    '01': 'SolarJan', '02': 'SolarFeb', '03': 'SolarMar', '04': 'SolarApr',
    '05': 'SolarMay', '06': 'SolarJun', '07': 'SolarJul', '08': 'SolarAug',
    '09': 'SolarSep', '10': 'SolarOct', '11': 'SolarNov', '12': 'SolarDec'
}

target_variable_Cos = {
    'FAO56': ['Distance_To_Ocean', 'Eastness', 'Elevation', 'Northness', 'Slope_Degree' , 'TWI'],
    'Mlake': ['Distance_To_Ocean', 'Eastness', 'Elevation', 'Northness', 'Slope_Degree' , 'TWI'],
    'Mwet':  ['Distance_To_Ocean', 'Eastness', 'Elevation', 'Northness', 'Slope_Degree' , 'TWI'],
    'Rain':  ['Distance_To_Ocean', 'Eastness', 'Elevation', 'Northness', 'Slope_Degree']
}

def get_nsw_season(month):
    if month in [12, 1, 2]: return "Summer"
    elif month in [3, 4, 5]: return "Autumn"
    elif month in [6, 7, 8]: return "Winter"
    else: return "Spring"

def safe_stratified_split(group, train_frac=0.70):
    n_total = len(group)
    if n_total == 1:
        n_train = 1
    else:
        n_train = int(round(n_total * train_frac))
        if n_train == n_total: n_train -= 1
        elif n_train == 0: n_train += 1
    return group.sample(n=n_train, random_state=42)

# ==================================================
# WORKER FUNCTION (Processes a single day)
# ==================================================
def run_evaluation_task(task_args):
    current_date, df_train_daily, vars_to_run = task_args
    day_str, month_str, year_str = current_date.split('/')
    season = get_nsw_season(int(month_str))
    
    daily_results = {var: [] for var in vars_to_run}
    Number_Of_Neighbours = 20
    coordinate_cols = ['Latitude', 'Longitude']

    # --- Nested Helper: Format rows ---
    def collect_result(var, model_setup_name, rmse, r2, pearson_r, nrmse, willmott_d, t_region):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def clean_metric(val):
            if pd.isna(val) or val == np.inf or val == -np.inf: return ""
            return float(val)

        def clean_int(val):
            try: return val.item() if hasattr(val, 'item') else int(val)
            except (ValueError, TypeError): return str(val)

        row = [
            timestamp, current_date, str(t_region), season, model_setup_name,
            clean_metric(rmse), clean_metric(r2), clean_metric(pearson_r),
            clean_metric(nrmse), clean_metric(willmott_d)
        ]
        daily_results[var].append(row)

    # --- Nested Helper: Calculate Metrics ---
    def evaluate_and_collect_by_region(var, model_name, y_true, y_pred, test_regions):
        df_eval = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'CMA_Name': test_regions})

        # By Region
        for region, group in df_eval.groupby('CMA_Name'):
            n_points = len(group)
            y_t, y_p = group['y_true'].values, group['y_pred'].values
            rmse = r2 = pearson_r = nrmse = willmott_d = np.nan

            if n_points >= 1:
                rmse = np.sqrt(mean_squared_error(y_t, y_p))
                mean_true = np.mean(y_t)
                if mean_true != 0: nrmse = rmse / mean_true
                num = np.sum((y_p - y_t)**2)
                den = np.sum((np.abs(y_p - mean_true) + np.abs(y_t - mean_true))**2)
                if den != 0: willmott_d = 1 - (num / den)

            if n_points >= 3:
                if np.var(y_t) > 0.01: r2 = r2_score(y_t, y_p)
                if np.std(y_t) > 0 and np.std(y_p) > 0: pearson_r = np.corrcoef(y_t, y_p)[0, 1]

            collect_result(var, model_name, rmse, r2, pearson_r, nrmse, willmott_d, region)

        # Statewide
        if len(y_true) > 1:
            full_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mean_full = np.mean(y_true)
            full_nrmse = full_rmse / mean_full if mean_full != 0 else np.nan
            num = np.sum((y_pred - y_true)**2)
            den = np.sum((np.abs(y_pred - mean_full) + np.abs(y_true - mean_full))**2)
            full_willmott = 1 - (num / den) if den != 0 else np.nan
            full_r2 = r2_score(y_true, y_pred)
            full_pearson = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
        else:
            full_rmse = full_r2 = full_pearson = full_nrmse = full_willmott = np.nan

        collect_result(var, model_name, full_rmse, full_r2, full_pearson, full_nrmse, full_willmott, "Full Dataset")

    # ==================================================
    # LOOP THROUGH REQUESTED VARIABLES FOR THIS DAY
    # ==================================================
    for var in vars_to_run:
        try:
            covariate_cols = target_variable_Cos[var] + [Month_dict[month_str]]
            
            # Drop NAs for this specific variable
            df_var = df_train_daily.dropna(subset=covariate_cols + [var, 'CMA_Name']).copy()
            if df_var.empty:
                continue

            # 1. Spatial Declustering
            agg_dict = {col: 'mean' for col in covariate_cols + [var] + coordinate_cols}
            agg_dict['CMA_Name'] = 'first'
            df_grouped = df_var.groupby('pixel_id').agg(agg_dict).reset_index()


            # 2. Train/Test Split & Scaling
            sampled_df = df_grouped.groupby('CMA_Name', group_keys=False).apply(safe_stratified_split, include_groups=False)
            
            train_df = df_grouped.loc[sampled_df.index]
            test_df = df_grouped.drop(sampled_df.index)

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            x_train, x_test = train_df[coordinate_cols].values, test_df[coordinate_cols].values
            y_train, y_test = train_df[var].values, test_df[var].values
            cma_test = test_df['CMA_Name'].values

            scaler = StandardScaler()
            p_train = scaler.fit_transform(train_df[covariate_cols])
            p_test = scaler.transform(test_df[covariate_cols])

            train_coords_verde = (x_train[:, 1], x_train[:, 0])
            test_coords_verde = (x_test[:, 1], x_test[:, 0])
            train_geodata_idw = np.column_stack([x_train[:, 1], x_train[:, 0], y_train])
            test_coords_idw = np.column_stack([x_test[:, 1], x_test[:, 0]])

            # ==================================================
            # 3A. TRADITIONAL MODELS
            # ==================================================
            try:
                idw_predictions = []
                for pt in test_coords_idw:
                    res_val = inverse_distance_weighting(
                        unknown_location=pt, known_locations=train_geodata_idw,
                        no_neighbors=Number_Of_Neighbours, power=2
                    )
                    idw_predictions.append(res_val)
                idw_preds = np.array(idw_predictions)
                evaluate_and_collect_by_region(var, 'IDW', y_test, idw_preds, cma_test)
            except Exception as e:
                pass

            try:
                lon_train = x_train[:, 1].reshape(-1, 1)
                lat_train = x_train[:, 0].reshape(-1, 1)
                X_train_gam = np.hstack([p_train, lon_train, lat_train])

                lon_test = x_test[:, 1].reshape(-1, 1)
                lat_test = x_test[:, 0].reshape(-1, 1)
                X_test_gam = np.hstack([p_test, lon_test, lat_test])

                n_covariates = p_train.shape[1]

                gam_terms = None
                for i, col_name in enumerate(covariate_cols):
                    if col_name == 'Elevation':
                        term = s(i, n_splines=8)
                    else:
                        term = l(i)

                    if gam_terms is None:
                        gam_terms = term
                    else:
                        gam_terms += term

                gam_terms += te(n_covariates, n_covariates + 1, n_splines=[15, 15])

                gam_model = LinearGAM(gam_terms)
                gam_model.gridsearch(X_train_gam, y_train)
                gam_preds = gam_model.predict(X_test_gam)

                evaluate_and_collect_by_region(var, 'pyGAM (Covariates + Spatial Spline)', y_test, gam_preds, cma_test)

            except Exception as e:
                pass


            variograms = ['spherical'] 

            for var_model in variograms:
                try:
                    ok_model = OrdinaryKriging(
                        x=x_train[:, 1], y=x_train[:, 0], z=y_train,
                        variogram_model=var_model, coordinates_type='geographic'
                    )
                    ok_preds, _ = ok_model.execute('points', x_test[:, 1], x_test[:, 0])
                    evaluate_and_collect_by_region(var, f"Ordinary Kriging + {var_model}", y_test, ok_preds, cma_test)

                except Exception as e:
                    pass

            models = {
                'SVM': SVR(C=1.0, gamma = 'scale'),
            }

            for model_name, model_instance in models.items():
                for var_model in variograms:
                    try:
                        rk = RegressionKriging(
                            regression_model=model_instance,
                            n_closest_points=Number_Of_Neighbours,
                            variogram_model=var_model,
                            coordinates_type='geographic',
                        )
                        rk.fit(p_train, x_train, y_train)
                        rk_preds = rk.predict(p_test, x_test)

                        setup_name = f"RK: {model_name} + {var_model}"
                        evaluate_and_collect_by_region(var, setup_name, y_test, rk_preds, cma_test)
                    except Exception as e:
                        pass

        except Exception as e:
            print(f"❌ Error processing {var} on {current_date}: {e}")

    # Return the dictionary back to the main thread
    return daily_results


# ==================================================
# MAIN EXECUTION BLOCK 
# ==================================================
if __name__ == '__main__':
    
    project_id = 'paleo-interpolation'
    client = bigquery.Client(project=project_id)
    print('Authenticated')

    # Local File Paths

    task_index = int(os.environ.get("BATCH_TASK_INDEX", 0))
    decade = task_index * 10  # Adjust this multiplier if your decades are numbered differently!
    
    OUTPUT_DIR = '/app/outputs/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching Covariates data...")
    query_covariants = """
            SELECT *
            FROM `paleo-interpolation.Paleo_Data.5km_Training_Data_CMA_Pixel`
        """
    df_train_covariates = client.query(query_covariants).to_dataframe()

    print(f"Fetching 3000 year run data...")
    query_3000_run = """
            SELECT Decade, Year
            FROM `paleo-interpolation.Paleo_Data.3000_Run_List`
        """
    df_3000_run = client.query(query_3000_run).to_dataframe()

    # --- 1. SETUP LOCAL CSV FILES ---
    variables_to_test = ['FAO56', 'Mlake', 'Mwet', 'Rain']
    csv_paths = {}
    headers = [
        "Timestamp", "Date", "CMA Region", "Season",
        "Model_Setup", "RMSE", "R_Squared", "Pearson_r",
        "NRMSE", "Willmott_d"
    ]

    for var in variables_to_test:
        file_path = os.path.join(OUTPUT_DIR, f"{var}_Evaluation_Results.csv")
        csv_paths[var] = file_path
        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as f:
                csv.writer(f).writerow(headers)

  

    
    years_to_process = df_3000_run[df_3000_run['Decade'] == decade]['Year'].values
    
    # --- Pre-calculate dates to filter out ones we've already processed ---
    sequential_dates = []
    for year in years_to_process:
        if year == 0 and decade == 0:
            start_date = date(year + 1, 1, 1)
            end_date = date(year + 1, 12, 31)
            total_days = (end_date - start_date).days + 1
            
            sequential_date = [
                (start_date + timedelta(days=offset)).strftime("%d/%m/%Y")[0:-1]+'0'
                for offset in range(total_days)
            ]
            sequential_dates.extend(sequential_date)
            
        else:
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)
            total_days = (end_date - start_date).days + 1
            
            sequential_date = [
                (start_date + timedelta(days=offset)).strftime("%d/%m/%Y")
                for offset in range(total_days)
                ]
            sequential_dates.extend(sequential_date)
    
    
    print(f"\n{'='*50}")
    print(f"BEGINNING EVALUATION PIPELINE FOR DECADE: {decade}s")
    print(f"{'='*50}")

    print(f"Fetching BigQuery data for {decade}s...")
    query = f"""
        SELECT Station_Region_ID, Decade, Year, Month, Day, FAO56, Mlake, Mwet, Rain
        FROM `paleo-interpolation.Paleo_Data.Paleo_aggregated_optimized`
        WHERE Decade = {decade}
    """
    df_bq_decade = client.query(query).to_dataframe()

    print("Merging BQ data with Local Covariates...")
    df_train_decade = pd.merge(df_train_covariates, df_bq_decade, on='Station_Region_ID')

    print("\n--- Partitioning Data for Multiprocessing ---")
    tasks = []
    # Notice we are only looping through the FILTERED sequential_dates
    for d in sequential_dates:
        day_str, month_str, year_str = d.split('/')
        df_train_daily = df_train_decade[
            (df_train_decade['Year'] == int(year_str)) &
            (df_train_decade['Month'] == int(month_str)) &
            (df_train_decade['Day'] == int(day_str))
        ].copy()

        if not df_train_daily.empty:
            tasks.append((d, df_train_daily, variables_to_test))

    print("Freeing up RAM...")
    del df_train_decade
    del df_bq_decade
    gc.collect()

    # --- 3. RUN MULTIPROCESSING (AS_COMPLETED) ---
    total_cores = multiprocessing.cpu_count()
    safe_cores = total_cores
    print(f"Firing up {safe_cores} CPU cores for processing...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
        futures = {executor.submit(run_evaluation_task, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            task_info = futures[future]
            d = task_info[0]
            try:
                daily_results = future.result() 
                for var, rows in daily_results.items():
                    if rows:
                        with open(csv_paths[var], mode='a', newline='') as f:
                            csv.writer(f).writerows(rows)
            except Exception as e:
                print(f"❌ Worker crashed for date {d}: {e}")

    # ==================================================
    # UPLOAD EVALUATIONS TO BIGQUERY
    # ==================================================
    print("\n--- Uploading Evaluation Results to BigQuery ---")
    
    import datetime
    
    # Helper function to bypass Pandas limitations using native Python dates
    def format_paleo_date(date_str):
        if pd.isna(date_str): 
            return None
        parts = str(date_str).split('/')
        if len(parts) == 3:
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2])
            # Return a native Python date object! 
            return datetime.date(year, month, day)
        return date_str

    for var in variables_to_test:
        csv_file = csv_paths[var]
        
        if os.path.exists(csv_file):
            table_id = f"{project_id}.Paleo_Data.{var}_3000_rerun"
            print(f"Preparing to upload {var} results to {table_id}...")

            try:
                df_upload = pd.read_csv(csv_file)
                
                if df_upload.empty:
                    print(f"⏭️ No data generated for {var}, skipping upload.")
                    continue

                # Apply the function to create actual datetime.date objects
                df_upload['Date'] = df_upload['Date'].apply(format_paleo_date)
                
                # Timestamp is safe because it's from the year 2026
                df_upload['Timestamp'] = pd.to_datetime(df_upload['Timestamp'])

                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND"
                )

                job = client.load_table_from_dataframe(df_upload, table_id, job_config=job_config)
                job.result() 

                print(f"✅ Successfully appended {len(df_upload)} rows to {table_id}.")
                
            except Exception as e:
                print(f"❌ Failed to upload {var} to BigQuery. Error: {e}")
        else:
            print(f"⚠️ Could not find CSV file for {var} at {csv_file}")

    print("\n🎉 All tasks complete. VM shutting down safely.")