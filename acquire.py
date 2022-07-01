import numpy as np
import pandas as pd
import os

def acquire_exoplanet_data():
    if os.path.isfile('exoplanet_data.csv'):
        df = pd.read_csv('exoplanet_data.csv', index_col=0)
    else:
        url_base = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query='
        query = 'select+hostname,pl_letter,sy_snum,sy_pnum,cb_flag,discoverymethod,disc_year,glat,glon,pl_orbper,sy_gaiamag,pl_controv_flag,pl_dens,pl_rade,pl_ratdor,pl_masse,st_teff,st_met,st_lum,st_logg,st_age,st_mass,st_dens,st_rad,sy_dist,rowupdate+from+ps'
        format = '&format=csv'
        url = url_base + query + format
        df = pd.read_csv(url)
        df.to_csv('exoplanet_data.csv')
    return df

