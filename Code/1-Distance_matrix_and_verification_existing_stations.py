
# This file is part of the article:
# "Leveraging Remote Traffic Data for Local Air Pollutant Estimation:
# A Scenario-Based Machine Learning Study Across London Monitoring Sites"
#
# Copyright (C) 2026 The authors
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import pandas as pd
import os
df = pd.read_csv("all_stations_more.csv")
df["Site Name"] = (
    df["Site Name"]
    .str.replace("\xa0", "", regex=False)
    .str.replace("Â", "", regex=False)
    .str.strip()
)
df["Sitename_type"] = df["Site Name"] + " " + df["Environment Type"]
# change to radians
coords = np.radians(df[["Latitude", "Longitude"]])

# Distance matrixs (in radians)
dist_matrix = haversine_distances(coords)

# change to km
earth_radius_km = 6371
dist_matrix_km = dist_matrix * earth_radius_km
dist_matrix_km = np.round(dist_matrix_km, 2)
dist_df = pd.DataFrame(
    dist_matrix_km,
    index=df["Sitename_type"],
    columns=df["Sitename_type"]
)

dist_df.to_csv("distance_matrix_km_btw_stations.csv", index=True)
stations = ['Camden Kerbside Urban Traffic','London Marylebone Road Urban Traffic','Westminster - Oxford Street Urban Traffic', 'Wandsworth - Putney High Street Urban Traffic', 'Camden - Euston Road Urban Traffic']

def find_nearest_by_type(dist_df, stations, suffix, k=2):
    """
    Find k nearest stations ending with a given suffix
    for each station in stations list.
    """
    results = {}
    candidates = [s for s in dist_df.index if s.endswith(suffix) and s not in stations]    
    for station in stations:        
        valid_candidates = [c for c in candidates if c != station] 
        distances = dist_df.loc[station, valid_candidates]   
        nearest = distances.sort_values().head(k)        
        results[station] = nearest    
    return results

nearest_background = find_nearest_by_type(
    dist_df,
    stations,
    suffix="Background",
    k=2
)
for i in nearest_background:
    print(f"Nearest background stations to {i}:")
    print(nearest_background[i])
    print("\n")

nearest_traffic = find_nearest_by_type(
    dist_df,
    stations,
    suffix="Traffic",
    k=2
)

for i in nearest_traffic:
    print(f"Nearest traffic stations to {i}:")
    print(nearest_traffic[i])
    print("\n")

dir_drive = r"Bkg_or_with_traffic\Neighbouring_stations"
files = ['Camden Kerbside 23jun2025-12dec2025_withtraffic.csv' ,'London Marylebone Road 23jun2025-12dec2025_withtraffic.csv',"Wandsworth - Putney High Street 23jun2025-12dec2025_withtraffic.csv"]
stations_considered = ['Camden Kerbside Urban Traffic','London Marylebone Road Urban Traffic','Westminster - Oxford Street Urban Traffic', 'Camden - Euston Road Urban Traffic','Wandsworth - Putney High Street Urban Traffic']
files_to_check = os.listdir(dir_drive)
dependant_variables = ['O3','NO', 'NO2', 'NOx as NO2', 'SO2', 'CO', 'PM10', 'PM25']


for n_station, station in enumerate(stations_considered):
    names_bkg = nearest_background[stations_considered[n_station]].index.tolist()
    clean_names_bkg = [name.split("Urban")[0].strip() for name in names_bkg]
    print(station, clean_names_bkg)
    for n_bks, back_station in enumerate(clean_names_bkg):
        flag = False
        for file_ in files_to_check:
            if back_station in file_:
                bkg_df = pd.read_csv(os.path.join(dir_drive, file_))
                bkg_df["date"] = pd.to_datetime(bkg_df["date"])
                pollutants_in = [col for col in bkg_df.columns if col in dependant_variables]
                pollutants_in.insert(0, "date")
                bkg_df = bkg_df[pollutants_in]
                bkg_df = bkg_df.rename(columns={col: f"{col}_bkg{n_bks}"  for col in bkg_df.columns if col != "date" })
                print(station, file_)
                flag=True
        if flag == False:
            print(f"No background station found for {station} and {back_station}")

for n_station, station in enumerate(stations_considered):
	names_traffic = nearest_traffic[stations_considered[n_station]].index.tolist()
	clean_names_trf = [name.split("Urban")[0].strip() for name in names_traffic]
	print(station, clean_names_trf)
	for n_trf, trf_station in enumerate(clean_names_trf):
		flag = False
		for file_ in files_to_check:
			if trf_station in file_:
				trf_df = pd.read_csv(os.path.join(dir_drive, file_))
				trf_df["date"] = pd.to_datetime(trf_df["date"])
				pollutants_in = [col for col in trf_df.columns if col in dependant_variables]
				pollutants_in.insert(0, "date")
				trf_df = trf_df[pollutants_in]
				trf_df = trf_df.rename(columns={col: f"{col}_trf{n_trf}"  for col in trf_df.columns if col != "date" })
				print(station, file_)
				flag=True
		if flag == False:
			print(f"No traffic station found for {station} and {trf_station}")
				
