import os
import glob
import shutil
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Google imports
from google.cloud import bigquery
from google.cloud import storage

# Kriging Imports
from pykrige.rk import RegressionKriging
from pykrige.ok import OrdinaryKriging

# Interpolators Imports
from pyinterpolate import inverse_distance_weighting

# PyGAM for Spline
from pygam import LinearGAM, l, te, s

# SKlearn and Model imports
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Standard Tools Imports
import gc # Garbage collector to free up memory
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import date, timedelta
import rasterio.features
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
import concurrent.futures
import multiprocessing


# ==================================================
# 1. SPATIAL INTERPOLATION FUNCTIONS
# ==================================================
def run_production_interpolation(current_date, target_variable, df_train_daily, df_grid, shapes, full_directory):

    day_str, month_str, year_str = current_date.split('/')
    print(f"\nProcessing Date: {current_date} | Variable: {target_variable}")

    Number_Of_Neighbours = 20
    models = ["RK_SVM_Spherical","Ordinary_Kriging", "IDW", 'Spline']

    models_to_run = []
    for name in models:
        file_name = f"{full_directory}{target_variable}_{name}_{year_str}_{month_str}_{day_str}.tif"
        if os.path.exists(file_name):
            print(f"  ⏭️ File already exists, skipping model: {name}")
        else:
            models_to_run.extend([name])

    if len(models_to_run) == 0:
        print(f"✅ All files for {current_date} already exist. Skipping day entirely.")
        return
    
    print(f"Running {models_to_run}")
    models = models_to_run

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
    
    covariate_cols = target_variable_Cos[target_variable] + [Month_dict[month_str]]

    agg_dict = {col: 'mean' for col in covariate_cols + [target_variable, 'Longitude', 'Latitude']}
    df_train_agg = df_train_daily.groupby('pixel_id').agg(agg_dict).reset_index()

    p_raw = df_train_agg[covariate_cols]
    scaler = StandardScaler()

    X_train = scaler.fit_transform(p_raw)
    X_grid = scaler.transform(df_grid[covariate_cols])

    y_train = df_train_agg[target_variable].values
    coords_train = df_train_agg[['Longitude', 'Latitude']].values
    coords_grid = df_grid[['Longitude', 'Latitude']].values

    res_x = 4994.196599428571062 
    res_y = 4993.14797156250097 

    min_lon, max_lon = df_grid['Original_Longitude'].min(), df_grid['Original_Longitude'].max()
    min_lat, max_lat = df_grid['Original_Latitude'].min(), df_grid['Original_Latitude'].max()

    width = int(np.round((max_lon - min_lon) / res_x)) + 1
    height = int(np.round((max_lat - min_lat) / res_y)) + 1
    transform = from_origin(west=min_lon - (res_x / 2), north=max_lat + (res_y / 2), xsize=res_x, ysize=res_y)
    nodata_value = -9999.0

    cols = np.round((df_grid['Original_Longitude'] - min_lon) / res_x).astype(int)
    rows = np.round((max_lat - df_grid['Original_Latitude']) / res_y).astype(int)

    nsw_mask = rasterio.features.geometry_mask(
        shapes, 
        transform=transform, 
        invert=False, 
        out_shape=(height, width)
    )

    for name in models:
        if name == "RK_SVM_Spherical":
            model = RegressionKriging(
                    regression_model=SVR(C=1.0, gamma='scale'),
                    variogram_model='spherical',
                    n_closest_points=Number_Of_Neighbours
                )
            model.fit(X_train, coords_train, y_train)
            preds = model.predict(X_grid, coords_grid)

        elif name == "Ordinary_Kriging":
            ok_model = OrdinaryKriging(
                x=coords_train[:, 0], y=coords_train[:, 1], z=y_train,
                variogram_model='spherical', coordinates_type='geographic'
            )
            ok_preds, _ = ok_model.execute('points', coords_grid[:, 0], coords_grid[:, 1])
            preds = np.array(ok_preds)

        elif name == "IDW":
            train_geodata_idw = np.column_stack([df_train_agg['Longitude'], df_train_agg['Latitude'], y_train])
            test_coords_idw = np.column_stack([df_grid['Longitude'], df_grid['Latitude']])
            idw_predictions = []
            for pt in test_coords_idw:
                res_val = inverse_distance_weighting(
                    unknown_location=pt,
                    known_locations=train_geodata_idw,
                    no_neighbors=Number_Of_Neighbours,
                    power=2
                )
                idw_predictions.append(res_val)

            preds = np.array(idw_predictions)

        elif name == "Spline":
            lon_train = coords_train[:, 0].reshape(-1, 1)
            lat_train = coords_train[:, 1].reshape(-1, 1)
            X_train_gam = np.hstack([X_train, lon_train, lat_train])

            lon_grid = coords_grid[:, 0].reshape(-1, 1)
            lat_grid = coords_grid[:, 1].reshape(-1, 1)
            X_grid_gam = np.hstack([X_grid, lon_grid, lat_grid])

            n_covariates = X_train.shape[1]
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

            lam_space = np.logspace(-3, 3, 3)
            gam_model = LinearGAM(gam_terms)
            gam_model.gridsearch(X_train_gam, y_train, lam=lam_space, progress=False)
            preds = gam_model.predict(X_grid_gam)
            
        np.round(preds, 2, out=preds)
        
        raster_array = np.full((height, width), nodata_value, dtype=np.float32)
        raster_array[rows, cols] = preds
        raster_array[nsw_mask] = nodata_value

        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "transform": transform,
            "count": 1,
            "dtype": raster_array.dtype,
            "crs": 'EPSG:8058',
            "nodata": nodata_value,
            "compress": "lzw", 
            "predictor": 3,
            "tiled": True      
        }
        
        file_name = f"{full_directory}{target_variable}_{name}_{year_str}_{month_str}_{day_str}.tif"

        with rasterio.open(file_name, "w", **out_meta) as final_dst:
            final_dst.write(raster_array, 1)
            
        del raster_array, preds
        gc.collect()

def run_pp_daily_task(task_args):
    d, df_train_daily, df_grid, shapes, full_directory = task_args
    try:
        run_production_interpolation(
            current_date=d, target_variable='Rain', df_train_daily=df_train_daily, 
            df_grid=df_grid, shapes=shapes, full_directory=full_directory
        )
    except Exception as e:
        print(f"❌ Failed on Date: {d}, Variable: Rain. Error: {e}")

# ==================================================
# 2. LOCAL AGGREGATION FUNCTIONS
# ==================================================
def sum_rasters_memory_safe(input_files, output_path):
    if not input_files: return

    valid_files = []
    lefts, bottoms, rights, tops = [], [], [], []
    
    for f in input_files:
        try:
            with rasterio.open(f) as src:
                bounds = src.bounds
                lefts.append(bounds.left)
                bottoms.append(bounds.bottom)
                rights.append(bounds.right)
                tops.append(bounds.top)
                valid_files.append(f)
        except Exception:
            print(f"    ❌ CORRUPT FILE (Skipping): {os.path.basename(f)}")

    if not valid_files: return
            
    intersect_left, intersect_bottom = max(lefts), max(bottoms)
    intersect_right, intersect_top = min(rights), min(tops)
    
    if intersect_left >= intersect_right or intersect_bottom >= intersect_top: return

    with rasterio.open(valid_files[0]) as src:
        meta = src.meta.copy()
        nodata_val = src.nodata if src.nodata is not None else -9999.0
        
        overlap_window = from_bounds(intersect_left, intersect_bottom, intersect_right, intersect_top, src.transform)
        overlap_window = overlap_window.round_offsets().round_shape()
        
        win_height, win_width = int(overlap_window.height), int(overlap_window.width)
        win_transform = rasterio.windows.transform(overlap_window, src.transform)
        
        meta.update({"height": win_height, "width": win_width, "transform": win_transform, "dtype": rasterio.float32})
        
        master_sum = np.zeros((win_height, win_width), dtype=np.float32)
        master_mask = np.zeros((win_height, win_width), dtype=bool)

    for f in valid_files:
        try:
            with rasterio.open(f) as src:
                file_window = from_bounds(intersect_left, intersect_bottom, intersect_right, intersect_top, src.transform)
                file_window = file_window.round_offsets().round_shape()
                arr = src.read(1, window=file_window)
                
                if arr.shape != (win_height, win_width):
                    min_h, min_w = min(arr.shape[0], win_height), min(arr.shape[1], win_width)
                    temp_arr = np.full((win_height, win_width), nodata_val, dtype=np.float32)
                    temp_arr[:min_h, :min_w] = arr[:min_h, :min_w]
                    arr = temp_arr
                
                valid_pixels = (arr != nodata_val)
                master_sum[valid_pixels] += arr[valid_pixels]
                master_mask |= valid_pixels
        except Exception:
            pass

    master_sum[~master_mask] = nodata_val

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(master_sum, 1)

def run_local_aggregations(output_folder, decade, years_to_process):
    print(f"\n--- Starting Local Aggregations for Decade {decade}s ---")
    
    for year in years_to_process:
        year_str = str(year)
        print(f"\n  Aggregating Year {year_str}...")

        daily_dir = os.path.join(output_folder, f'{decade}/{year}/Daily_Rain/')
        monthly_dir = os.path.join(output_folder, f'{decade}/{year}/Monthly_Rain/')
        yearly_dir = os.path.join(output_folder, f'{decade}/{year}/Yearly_Rain/')

        os.makedirs(monthly_dir, exist_ok=True)
        os.makedirs(yearly_dir, exist_ok=True)

        monthly_groups = defaultdict(lambda: defaultdict(list))
        
        # 1. Group the Daily files generated by the interpolation block
        if os.path.exists(daily_dir):
            for filename in os.listdir(daily_dir):
                if not filename.endswith('.tif'): continue
                
                # Filename logic: Rain_RK_SVM_Spherical_2000_01_01.tif
                parts = filename.replace('.tif', '').split('_')
                month_part = parts[-2]
                model_name = "_".join(parts[1:-3]) 
                
                local_path = os.path.join(daily_dir, filename)
                monthly_groups[model_name][month_part].append(local_path)

        # 2. Process Monthly, then Yearly
        for model, months in monthly_groups.items():
            yearly_files = [] 
            
            for month, files in sorted(months.items()):
                month_filename = f"Rain_{model}_{year_str}_{month}_Total.tif"
                monthly_local_path = os.path.join(monthly_dir, month_filename)
                
                # Create monthly sum (Skip if already done on a previous interrupted run)
                if not os.path.exists(monthly_local_path):
                    sum_rasters_memory_safe(files, monthly_local_path)
                
                if os.path.exists(monthly_local_path):
                    yearly_files.append(monthly_local_path)
            
            if yearly_files:
                year_filename = f"Rain_{model}_{year_str}_Total.tif"
                yearly_local_path = os.path.join(yearly_dir, year_filename)
                print(f"    -> Creating Yearly Total: {year_filename}")
                
                if not os.path.exists(yearly_local_path):
                    sum_rasters_memory_safe(yearly_files, yearly_local_path)

    print("\n✅ Local Aggregations Complete!")

# ==================================================
# 3. GCP UPLOAD FUNCTION
# ==================================================
def upload_outputs_to_gcp(local_folder, bucket_name, gcs_prefix, storage_client):
    print(f"\n--- Starting Upload to GCP Bucket: {bucket_name} ---")
    bucket = storage_client.bucket(bucket_name)
    
    # Walk through all directories and files in the local output folder
    # This will now automatically include Daily_Rain, Monthly_Rain, and Yearly_Rain
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            if file.endswith('.tif'):
                local_path = os.path.join(root, file)
                # Maintain the folder structure
                relative_path = os.path.relpath(local_path, local_folder)
                blob_path = f"{gcs_prefix}/{relative_path}"
                
                blob = bucket.blob(blob_path)
                print(f"Uploading {file} to gs://{bucket_name}/{blob_path} ...")
                blob.upload_from_filename(local_path)
    print("✅ All files uploaded successfully!")


# ==================================================
# MAIN EXECUTION BLOCK 
# ==================================================
if __name__ == '__main__':
    
    project_id = 'paleo-interpolation'
    client = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)
    print('Authenticated')

    output_folder = '/app/outputs/'

    task_index = int(os.environ.get("BATCH_TASK_INDEX", 0))
    decade = task_index * 10 

    print(f"Fetching 3000 year run data...")
    query_3000_run = """
            SELECT Decade, Year
            FROM `paleo-interpolation.Paleo_Data.3000_Run_List`
        """
    df_3000_run = client.query(query_3000_run).to_dataframe()

    print(f"Fetching Covariates data...")
    query_covariants = """
            SELECT *
            FROM `paleo-interpolation.Paleo_Data.5km_Training_Data_CMA_Pixel`
        """
    df_train_covariates = client.query(query_covariants).to_dataframe()

    print(f"Fetching Grid data...")
    query_covariants = """
            SELECT *
            FROM `paleo-interpolation.Paleo_Data.Base_Interpolation_Data`
        """
    df_grid = client.query(query_covariants).to_dataframe().sort_values(['Latitude', 'Longitude'], ascending=[False, True]).dropna()

    print("Loading NSW Boundary...")
    bucket = storage_client.bucket('dcceew-input-data')
    blob = bucket.blob('Vector_Data/NSW_Boundary/NSW_Poly.geojson')
    NSW_blob_bytes = blob.download_as_bytes()

    NSW_gdf = gpd.read_file(NSW_blob_bytes).to_crs("EPSG:8058")
    shapes = NSW_gdf.geometry.values

    variables_to_test = ['Rain']
    years_to_process = df_3000_run[df_3000_run['Decade'] == decade]['Year'].values

    print(f"\n{'='*50}")
    print(f"BEGINNING FULL PIPELINE FOR DECADE: {decade}s (Batch Task Index: {task_index})")
    print(f"{'='*50}")

    print(f"Fetching BigQuery data for {decade}s...")
    query = f"""
        SELECT Station_Region_ID, Decade, Year, Month, Day, Rain
        FROM `paleo-interpolation.Paleo_Data.Paleo_aggregated_optimized`
        WHERE Decade = {decade} And Rain IS NOT NULL
    """
    df_rain_decade = client.query(query).to_dataframe().dropna()

    print("Merging BQ Rain data with Local Covariates...")
    df_train_decade = pd.merge(df_train_covariates, df_rain_decade, on='Station_Region_ID')

    sequential_dates_three_years = []
    for year in years_to_process:
        if year == 0 and decade == 0:
            start_date = date(year + 1, 1, 1)
            end_date = date(year + 1, 12, 31)
            total_days = (end_date - start_date).days + 1
            
            sequential_dates = [
                (start_date + timedelta(days=offset)).strftime("%d/%m/%Y")[0:-1]+'0'
                for offset in range(total_days)
            ]
            sequential_dates_three_years.extend(sequential_dates)
            
        else:
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)
            total_days = (end_date - start_date).days + 1
            
            sequential_dates = [
                (start_date + timedelta(days=offset)).strftime("%d/%m/%Y")
                for offset in range(total_days)
                ]
            sequential_dates_three_years.extend(sequential_dates)
    
    decade_directory = os.path.join(output_folder, f'{decade}/')
    os.makedirs(decade_directory, exist_ok=True)
    
    for year in years_to_process:
        year_directory = os.path.join(output_folder, f'{decade}/{year}/')
        daily_directory = os.path.join(output_folder, f'{decade}/{year}/Daily_Rain/')
        os.makedirs(year_directory, exist_ok=True)
        os.makedirs(daily_directory, exist_ok=True)

    print("\n--- Partitioning Data for Multiprocessing ---")
    tasks = []
    for d in sequential_dates_three_years:
        day_str, month_str, year_str = d.split('/')
        daily_directory = os.path.join(output_folder, f'{decade}/{int(year_str)}/Daily_Rain/')
        df_train_daily = df_train_decade[
            (df_train_decade['Year'] == int(year_str)) &
            (df_train_decade['Month'] == int(month_str)) &
            (df_train_decade['Day'] == int(day_str))
        ].copy()

        if not df_train_daily.empty:
            tasks.append((d, df_train_daily, df_grid, shapes, daily_directory))

    print("Freeing up RAM...")
    del df_train_decade
    del df_rain_decade
    gc.collect()

    print("\n--- Starting Daily Interpolations ---")
    
    total_cores = multiprocessing.cpu_count()
    safe_cores = max(1, int(total_cores * 0.65)) 
    print(f"Firing up {safe_cores} CPU cores for processing...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
        results = list(executor.map(run_pp_daily_task, tasks))

    print("\n--- Verifying Completeness (The Once-Over) ---")
    
    missing_tasks = []
    for task in tasks:
        current_date = task[0]
        specific_daily_directory = task[4] 
        day_str, month_str, year_str = current_date.split('/')
        date_identifier = f"_{year_str}_{month_str}_{day_str}.tif"
        
        # Check if the specific daily directory exists and contains the required file
        if os.path.exists(specific_daily_directory):
            existing_files = set(os.listdir(specific_daily_directory))
            if not any(date_identifier in f for f in existing_files):
                missing_tasks.append(task)
        else:
            missing_tasks.append(task)

    if missing_tasks:
        print(f"⚠️ Found {len(missing_tasks)} missing/failed dates! Spinning up rescue workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
            list(executor.map(run_pp_daily_task, missing_tasks))
        print("✅ Rescue run complete!")
    else:
        print("✅ Perfect run! All dates successfully processed.")

    print(f"\n--- Completed Interpolations for decade {decade}s ---")

    # ==================================================
    # NEW: PROCESS LOCAL AGGREGATIONS (MONTHLY & YEARLY)
    # ==================================================
    run_local_aggregations(output_folder, decade, years_to_process)

    # ==================================================
    # TRIGGER GCP BATCH UPLOAD (Uploads Everything)
    # ==================================================
    bucket_name = 'dcceew-input-data'
    gcs_prefix = 'Geotiff_3000_Rerun' # The master folder inside the bucket
    
    upload_outputs_to_gcp(output_folder, bucket_name, gcs_prefix, storage_client)