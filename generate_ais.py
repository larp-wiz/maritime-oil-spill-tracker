import pandas as pd

spill_x = 220650.00
spill_y = 2719350.02

# Vessel fleet with behavioral metrics following Marine Cadastre AIS standards
vessel_data = [
    {
        "MMSI": "419001234", 
        "VesselName": "MT ARABIAN STAR", 
        "VesselType": "Oil Tanker", 
        "X": spill_x + 150, 
        "Y": spill_y + 200, 
        "SOG_knots": 1.8,            # Low speed anomaly (Loitering/Dumping)
        "Draft_m": 12.5,
        "NavStatus": "Moored / Slowed"
    },
    {
        "MMSI": "211394080", 
        "VesselName": "INDEPENDENCE II", 
        "VesselType": "Cargo Ship", 
        "X": spill_x + 4500, 
        "Y": spill_y + 3100, 
        "SOG_knots": 14.2,           # Normal transit speed
        "Draft_m": 8.0,
        "NavStatus": "Underway using engine"
    },
    {
        "MMSI": "563029110", 
        "VesselName": "SEAWAVE EXPRESS", 
        "VesselType": "Container Ship", 
        "X": spill_x - 8200, 
        "Y": spill_y + 1200, 
        "SOG_knots": 18.5,           # Normal transit speed
        "Draft_m": 10.2,
        "NavStatus": "Underway using engine"
    },
    {
        "MMSI": "311009820", 
        "VesselName": "GLOBAL TRANSPORT", 
        "VesselType": "Bulk Carrier", 
        "X": spill_x + 300, 
        "Y": spill_y - 400, 
        "SOG_knots": 15.0,           # Nearby, but high speed (transiting)
        "Draft_m": 9.1,
        "NavStatus": "Underway using engine"
    }
]

df = pd.DataFrame(vessel_data)
df.to_csv("marine_cadastre_ais.csv", index=False)
print(" Created marine_cadastre_ais.csv with behavioral metrics!")