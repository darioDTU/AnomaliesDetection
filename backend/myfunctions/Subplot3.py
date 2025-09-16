import numpy as np
import matplotlib.pyplot as plt

def Subplot3(datas, keys, datas_available, columns_index):
    """
    Plot differents data evolution according to depth graphs

    In:
        -> Datas to plot
        -> Keys to uniques values
        -> Kind of datas available/wanted to be plot
    Out:
        -> Plot of datas according to depth
    """
    d = {}
    latitude_index = columns_index['LATITUDE']
    for i in range(len(keys)):
        mask = datas[:, latitude_index] == keys[i]
        d[i] = datas[mask]
    keys = d.keys()
    # Number of plots is based on the length of datas_available
    n_plots = len(datas_available)
    if n_plots < 4:
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 50), dpi=100)
    elif n_plots > 3 and n_plots < 7:
        rows = 2
        cols = int(np.ceil(n_plots / 2))
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 50 * rows), dpi=100)
    elif n_plots > 6 and n_plots < 10:
        rows = 3
        cols = int(np.ceil(n_plots / 3))
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 50 * rows), dpi=100)
    else:
        rows = 4
        cols = int(np.ceil(n_plots / 4))
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 50 * rows), dpi=100)
    plt.subplots_adjust(wspace=0.3)

    # This will ensure it works even if there's only one subplot
    if n_plots != 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Set a universal title
    fig.suptitle(f'Data Evolution with Depth for {datas[0, columns_index['PLATFORM_NUMBER']]}')
    # Plot each data type in its respective subplot
    for idx, data_type in enumerate(datas_available):
        ax = axes[idx]
        for i in range(len(keys)):
            try:
                # Attempt to convert the data column to float for finite checking
                data_column = d[i][:, columns_index[data_type]].astype(float)
                finite_mask = np.isfinite(data_column)
            except TypeError:
                print(f"Data type conversion error for {data_type}. Likely non-numeric data present.")
                continue  # Skip this iteration if conversion fails

            if not np.any(finite_mask):
                print(f"No valid data available for plotting in subplot {idx} for key {i}")
                continue

            # Proceed with plotting only if there are finite values
            ax.plot(data_column[finite_mask], d[i][finite_mask, columns_index['PRES']], label=f'Lon: {d[i][0, columns_index["LONGITUDE"]]}, Lat: {d[i][0, columns_index["LATITUDE"]]}')
            if data_type == 'CHLA':
                ax.set_xlabel(f'{data_type} (mg/m3)')
                ax.set_title('Chlorophyll-a evolution')
            elif data_type == 'TEMP':
                ax.set_xlabel(f'{data_type} (°C)')
                ax.set_title('Temperature evolution')
            elif data_type == 'PSAL':
                ax.set_xlabel(f'{data_type} (g/kg)')
                ax.set_title('Salinity evolution')
            elif data_type == 'DOXY':
                ax.set_xlabel(f'{data_type} (hpa)')
                ax.set_title('Oxygen evolution')
            elif data_type == 'NITRATE':
                ax.set_xlabel(f'{data_type} (mg/L)')
                ax.set_title('Nitrate evolution')
            elif data_type == 'BBP700':
                ax.set_xlabel(f'{data_type} (m-1)')
                ax.set_title('Backscatter evolution')
            elif data_type == 'DOWN_IRRADIANCE412':
                ax.set_xlabel(f'{data_type} (W.m-1.nm-1)')
                ax.set_title('Downwelling Irradiance 412 evolution')
            elif data_type == 'DOWN_IRRADIANCE443':
                ax.set_xlabel(f'{data_type} (W.m-1.nm-1)')
                ax.set_title('Downwelling Irradiance 443 evolution')
            elif data_type == 'PH_IN_SITU_TOTAL':
                ax.set_xlabel(f'{data_type} (PH)')
                ax.set_title('PH in situ evolution')
            elif data_type == 'CDOM':
                ax.set_xlabel(f'{data_type} (m-1)')
                ax.set_title('Colored Dissolved Organic Matter evolution')
        ax.set_ylabel('Depth')
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend()
    
    plt.show()