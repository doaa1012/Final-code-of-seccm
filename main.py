import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from nptdms import TdmsFile
from datatree import DataTree
import matplotlib.pyplot as plt


def load_tdms(path: str | Path, ph: float, d_cap: float, offset_pot: float = 0, 
              st_pot: float = 0.21, apr_shift: int = 0) -> DataTree | None:
    """
    Loads a TDMS file from a given path, processes it into a DataTree structure, and computes the current density.

    :param path: string or Path object to the TDMS file
    :param ph: pH of the electrolyte
    :param d_cap: diameter of the capillary in nm, used to obtain the current density
    :param offset_pot: potential difference of the used reference electrode to the standard reference electrode in V
    :param st_pot: standard potential of the (standard) reference electrode in V, for Ag|AgCl|3M KCl it is 0.21 V, default value
    :param apr_shift: Number of additional indices to consider before the first approach point
    :return: DataTree with the experiment data structured for further analysis
    """
    
    # Load the TDMS file and convert the data into a DataFrame
    tdms_file = TdmsFile(path)
    data = tdms_file['Data'].as_dataframe()

    # Add a column for the cumulated time
    data['t(s)'] = data['dt(s)'].cumsum()

    # Return None if invalid data (negative line numbers or missing retraction tip)
    if (data['Line Number'] < 0).any().any() or 3 not in data['FeedbackType '].values:
        return None

    # Convert line numbers to integers
    data['Line Number'] = data['Line Number'].astype(int)

    # Group the data by line numbers and compute the average feedback type (FT)
    feed_types = data.groupby('Line Number')['FeedbackType '].mean()

    # Determine valid measurements: retraction (FT 3), approach (FT 2), and in-between movement points are excluded
    is_meas = ~ ((feed_types == 3) | (feed_types == 3).shift(periods=1, fill_value=False) | (feed_types == 2))

    # Handle case where no valid first approach is found (FT 1)
    if feed_types[feed_types == 1].empty:
        return None

    # Find the index of the first approach and set all measurements before that as False
    apr1_idx = feed_types[feed_types == 1].index[-1]
    is_meas[:apr1_idx+apr_shift] = False

    # Determine measurement areas and associate them with line numbers
    apr = (feed_types == 1).shift(periods=1, fill_value=False) | (feed_types == 2)
    meas1 = apr.shift(periods=1, fill_value=False)
    meas_area = meas1.astype(int).cumsum()
    meas_area[~is_meas] = 0

    # Assign measurement area numbers back to the DataFrame
    data['MeasNumber'] = meas_area[data['Line Number']].reset_index(drop=True)

    # Compute the sweep number by resetting cumulative sum after every NaN value (not measuring)
    meas_nan = is_meas.astype(int).replace(0, np.nan)
    meas_cum = meas_nan.cumsum().ffill().fillna(0)
    meas_restart = meas_cum.mask(is_meas, np.nan).ffill()
    n_sweeps = (meas_cum - meas_restart).astype(int)
    data['SweepNumber'] = n_sweeps[data['Line Number']].reset_index(drop=True)

    # Remove data points where the line number jumps by more than one
    data = data[data['Line Number'].diff() <= 1]

    # Filter out non-measurement points and keep only relevant columns
    keep_cols = ['MeasNumber', 'SweepNumber', 'X (um)', 'Y (um)', 'V1 (V)', 'Current1 (A)', 't(s)']
    d_filt = data[data['MeasNumber'] != 0][keep_cols].reset_index(drop=True)

    # Convert the potential to the reference electrode using the Nernst equation
    d_filt['Potential'] = d_filt['V1 (V)'] + offset_pot + st_pot + 0.059 * ph

    # Calculate the area of the capillary in cm^2
    cap_area = np.pi * (d_cap * 1e-7 / 2) ** 2

    # Convert the current to a current density in A/cm^2
    d_filt['CurrDens'] = d_filt['Current1 (A)'] / cap_area

    # Convert the DataFrame into xarray.Dataset for DataTree structure
    ds = xr.Dataset(
        {
            'Potential': (['time'], d_filt['Potential']),
            'CurrentDensity': (['time'], d_filt['CurrDens']),
            'X_um': (['time'], d_filt['X (um)']),
            'Y_um': (['time'], d_filt['Y (um)']),
            'SweepNumber': (['time'], d_filt['SweepNumber']),
            'MeasNumber': (['time'], d_filt['MeasNumber']),
        },
        coords={
            'time': d_filt['t(s)']
        }
    )

    # Create the DataTree from the xarray.Dataset
    tree = DataTree(ds)

   # Add metadata
    tree.attrs['filename'] = Path(path).name  # Name of the file
    tree.attrs['pH'] = ph  # pH value of the electrolyte
    tree.attrs['capillary_diameter_nm'] = d_cap  # Diameter of the capillary in nanometers
    tree.attrs['capillary_area_cm2'] = cap_area  # Computed area of the capillary in cm²
    tree.attrs['offset_potential_V'] = offset_pot  # Offset potential in volts
    tree.attrs['standard_potential_V'] = st_pot  # Standard potential in volts
    tree.attrs['approach_shift'] = apr_shift  # Shift value for approach determination
    tree.attrs['num_sweeps'] = d_filt['SweepNumber'].nunique()  # Number of sweeps

    return tree

def extract_lsvs(tree: DataTree, sweep: int, rem_hop_areas: int | list[int] = None, pots: list | np.ndarray = None) -> DataTree | None:
    """
    Extracts LSVs of all hopping areas from a given sweep number, interpolates them, and computes the average LSV.
    
    :param tree: DataTree with the data from the experiment
    :param sweep: Sweep number to extract from the voltammogram
    :param rem_hop_areas: List of hopping areas to be removed from the LSVs (optional)
    :param pots: List or numpy array of potentials to interpolate to (optional)
    :return: DataTree with interpolated LSVs and averaged LSV across hopping areas
    """
    # Access the dataset from DataTree
    ds = tree.ds

    # Filter the dataset for the given sweep number
    d_sweep = ds.where(ds.SweepNumber == sweep, drop=True)

    # Remove specific hopping areas if defined
    if rem_hop_areas is not None:
        rem_hop_areas = [rem_hop_areas] if isinstance(rem_hop_areas, int) else rem_hop_areas
        d_sweep = d_sweep.where(~d_sweep.MeasNumber.isin(rem_hop_areas), drop=True)

    # If no potentials list is provided, calculate the mutual minimum and maximum potentials
    if pots is None:
        s_min, s_max = d_sweep.Potential.min().item(), d_sweep.Potential.max().item()

        # Calculate the scan rate by finding the most frequent difference between potentials
        s_rates = []
        for area in np.unique(d_sweep.MeasNumber.values):
            potentials = d_sweep.where(d_sweep.MeasNumber == area, drop=True).Potential.values
            diffs = np.diff(potentials)
            if len(diffs) > 0:
                s_rates.extend(diffs)

        # Find the most frequent scan rate (mode)
        s_rate_mode = -pd.Series(s_rates).mode()

        # If no frequent scan rate is found, return None
        if len(s_rate_mode) == 0:
            return None

        # Use the mode scan rate to define the potential range for interpolation
        s_rate = s_rate_mode.iloc[0]
        pots = np.arange(s_min, s_max, s_rate)

    # Prepare to store the interpolated LSVs
    areas = np.unique(d_sweep.MeasNumber.values)
    interp_data = {}

    # Interpolate the LSVs for each hopping area individually
    for h_area in areas:
        d = d_sweep.where(d_sweep.MeasNumber == h_area, drop=True).sortby('Potential')
        interp_data[h_area] = np.interp(pots, d.Potential.values, d.CurrentDensity.values, left=np.nan, right=np.nan)



    # Create an xarray.Dataset for the interpolated data
    lsv_ds = xr.Dataset(
        data_vars={f'Area_{area}': (['Potential'], interp_data[area]) for area in areas},
        coords={'Potential': pots}
    )
    # Drop rows where all columns are NaN
    lsv_ds = lsv_ds.dropna(dim="Potential", how="all")

    # Optional: Interpolate missing values for partial NaN rows
    lsv_ds = lsv_ds.interpolate_na(dim="Potential", method="linear", fill_value="extrapolate")


    # Compute the average LSV across all areas
    avg_lsv = np.nanmean(list(interp_data.values()), axis=0)
    lsv_ds['Average Current density [A/cm^2]'] = ('Potential', avg_lsv)

    # Create a new DataTree for the interpolated LSVs and add it to the existing tree
    lsv_tree = DataTree(lsv_ds)
    lsv_tree.attrs['sweep'] = sweep
    lsv_tree.attrs['removed_hop_areas'] = rem_hop_areas
    lsv_tree.attrs['interpolated_potentials'] = pots

    return lsv_tree