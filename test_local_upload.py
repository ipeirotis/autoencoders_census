from dataset.loader import DataLoader
# 1. Simulate an upload by reading a local file as raw bytes
# (This is exactly what the Google Cloud library will give you later)
with open("data/sadc_2015only_national.csv", "rb") as f:
    fake_uploaded_bytes = f.read()

print(f"Simulated upload size: {len(fake_uploaded_bytes)} bytes")

# 2. Initialize your loader
loader = DataLoader(
    drop_columns=[], 
    rename_columns={}, 
    columns_of_interest=[]
)

# 3. Test the new function!
# If this works, your loader is ready for the cloud.
df, types = loader.load_uploaded_csv(fake_uploaded_bytes)

print("\nSuccess!")
print(f"Processed DataFrame Shape: {df.shape}")
print("First 5 columns:", list(df.columns[:5]))