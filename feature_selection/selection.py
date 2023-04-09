from typing import Tuple
import pandas as pd

def get_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df = df.drop(["RefID", "PurchDate", "VehYear", "Trim", "Color",
             "WheelTypeID", "WheelType", "Nationality", "Size",
             "TopThreeAmericanName", "BYRNO", "VNST", "WarrantyCost"])
 
    X = df.drop(["IsBadBuy"]) 
    Y = df.filter(["IsBadBuy"])

    return X, Y
