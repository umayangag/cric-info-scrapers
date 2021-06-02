def fill_missing_attributes(data_df):
    final_df = data_df.copy()
    for column_name in ["bowling_temp", "bowling_wind", "bowling_rain", "bowling_humidity", "bowling_cloud",
                        "bowling_pressure", "bowling_viscosity"]:
        final_df[column_name] = final_df[column_name].fillna(final_df[column_name].mean())
    return final_df.fillna(0)
