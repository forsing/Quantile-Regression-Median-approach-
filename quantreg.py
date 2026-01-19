# Quantile Regression (Median approach) for Lottery Prediction


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/Users/milan/Desktop/GHQ/data/loto7hh_4548_k5.csv')
# 4548 historical draws of Lotto 7/39 (Serbia)




def predict_next_with_quantreg(df):
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    predictions = {}
    
    # Create a simple lag feature (previous draw) for each column
    for col in cols:
        df[f'{col}_lag'] = df[col].shift(1)
    
    # Drop first row with NaN lag
    df_model = df.dropna()
    
    for col in cols:
        # Define the formula: current number ~ previous number
        formula = f"{col} ~ {col}_lag"
        
        # Fit Quantile Regression (tau=0.5 is the median)
        model = smf.quantreg(formula, df_model)
        res = model.fit(q=0.5)
        
        # Get the last known value to predict the next
        last_val = df[col].iloc[-1]
        
        # Predict next value
        # Prediction = Intercept + Coef * last_val
        pred_val = res.params['Intercept'] + res.params[f'{col}_lag'] * last_val
        
        # Round and ensure it's within reasonable bounds (lottery numbers are positive)
        predictions[col] = max(1, int(round(pred_val)))
        
    return predictions

# Generate prediction
quantile_preds = predict_next_with_quantreg(df_raw)

# Create dataframe for display
quantreg_df = pd.DataFrame([quantile_preds])
# quantreg_df.index = ['Quantile Regression Prediction (Median)']

print()
print("Lottery prediction generated using Quantile Regression (Median approach).")
print()


print()
print("Quantile Regression Prediction (Median) Results:")
print(quantreg_df.to_string(index=True))
print()
"""
Quantile Regression Prediction (Median) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     4     9    15    20    25    31    36
"""





"""

QSVR 
Quantum Data Re-uploading Regression 
Multi-Qubit VQR 
QRC 
QNN 
QCNN 
QKA 
QRNN 
QMTR 
QGBR 
QBR 
QSR 






QCM

QDR 

QELM

QGPR 

QTL




quantile 

VQC

"""



"""
ok for VQC and QSVR and Quantum Data Re-uploading Regression and Multi-Qubit VQR and QRC and QNN and QCNN and QKA and QRNN and QMTR and QGBR and QBR and QSR and QDR and QGPR and QTL and QELM, give next model quantum regression with qiskit
"""