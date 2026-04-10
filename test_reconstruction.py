
import pandas as pd
import logging
from features.transform import Table2Vector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reconstruction_logic():
    # 1. Create Dummy Data
    data = {
        "col1": ["A", "B", "A", "C"],
        "col2": ["X", "Y", "Y", "X"]
    }
    df = pd.DataFrame(data)
    logger.info("Original DataFrame:")
    print(df)

    # 2. Initialize Vectorizer
    model_variable_types = {col: "categorical" for col in df.columns}
    vectorizer = Table2Vector(model_variable_types)

    # 3. Vectorize
    vectorized_df = vectorizer.vectorize_table(df)
    logger.info("\nVectorized DataFrame:")
    print(vectorized_df)

    # 4. Simulate Model Reconstruction (just using the same data for perfect reconstruction, 
    # but let's change one slightly to simulate error)
    vectorized_df.values.copy()
    
    # Let's say for row 1 (B, Y), the model predicts (A, Y)
    # col1 has categories A, B, C. A is index 0, B is index 1.
    # We find the columns for col1
    col1_cols = [c for c in vectorized_df.columns if "col1" in c]
    print(f"\nCol1 columns: {col1_cols}")
    
    # 0 -> A, 1 -> B, 2 -> C (usually sorted)
    # Row 1 is B (so [0, 1, 0]). We change it to predict A ([1, 0, 0])
    # But wait, we need to know the index mapping.
    
    # Let's just create a dummy reconstruction df directly
    reconstruction_df = vectorized_df.copy()
    
    # Let's mess up the second row
    # Assume col1__A is the first column.
    reconstruction_df.iloc[1, 0] = 0.9 # High prob for A
    reconstruction_df.iloc[1, 1] = 0.1 # Low prob for B
    
    logger.info("\nSimulated Reconstruction Probabilities (Row 1 changed):")
    print(reconstruction_df)

    # 5. Tabularize (Reverse Transform)
    reconstructed_table = vectorizer.tabularize_vector(reconstruction_df)
    logger.info("\nReconstructed Table:")
    print(reconstructed_table)
    
    # 6. Compare and format
    # Join with original to see delta
    comparison = df.copy()
    for col in df.columns:
        # Create a combined string "Original -> Reconstructed" if different
        comparison[col + "_rec"] = reconstructed_table[col]
        
        comparison[col + "_display"] = comparison.apply(
            lambda row: f"{row[col]} -> {row[col+'_rec']}" if row[col] != row[col+'_rec'] else row[col],
            axis=1
        )

    logger.info("\nComparison for Display:")
    print(comparison[[c + "_display" for c in df.columns]])

if __name__ == "__main__":
    test_reconstruction_logic()
