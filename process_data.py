import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg
from pyspark.sql.window import Window

RAW_DIR = 'data_raw'
PROCESSED_DIR = 'data_processed'
RAW_FILE = os.path.join(RAW_DIR, 'dummy_sensor_data.csv')
PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'processed_data.parquet')

# Generate dummy data if data_raw is empty
def generate_dummy_data():
    if not os.listdir(RAW_DIR):
        n = 1000
        np.random.seed(42)
        df = pd.DataFrame({
            'asset_id': np.random.choice(['A', 'B', 'C'], size=n),
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'sensor_1': np.random.normal(50, 10, size=n),
            'sensor_2': np.random.normal(100, 20, size=n),
            'sensor_3': np.random.normal(75, 5, size=n),
        })
        # Add some nulls
        for colname in ['sensor_1', 'sensor_2', 'sensor_3']:
            df.loc[df.sample(frac=0.05).index, colname] = np.nan
        df.to_csv(RAW_FILE, index=False)
        print(f"Dummy data generated at {RAW_FILE}")

def main():
    generate_dummy_data()
    spark = SparkSession.builder.appName('ProcessData').getOrCreate()
    df = spark.read.csv(RAW_FILE, header=True, inferSchema=True)

    # Clean nulls (fill with mean)
    for c in ['sensor_1', 'sensor_2', 'sensor_3']:
        mean_val = df.select(avg(col(c))).first()[0]
        df = df.na.fill({c: mean_val})

    # Parse timestamp
    df = df.withColumn('timestamp', to_timestamp(col('timestamp')))

    # Rolling average feature (window of 5)
    window = Window.partitionBy('asset_id').orderBy('timestamp').rowsBetween(-2, 2)
    df = df.withColumn('sensor_1_rollavg', avg(col('sensor_1')).over(window))

    # Write to Parquet
    df.write.mode('overwrite').parquet(PROCESSED_FILE)
    print(f"Processed data written to {PROCESSED_FILE}")
    spark.stop()

if __name__ == '__main__':
    main() 