import pandas as pd

# Count buildings
def get_buildings_with_high_meter(df, threshold):
    """
   Return the building ids with a count of meter readings above a certain threshold.
   """
    return set(df.meter_reading > threshold)
