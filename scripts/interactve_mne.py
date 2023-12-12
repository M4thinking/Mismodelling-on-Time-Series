import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import mne, re, warnings, os

def compute_labels():
    # Load cvs
    experts = ['A', 'B', 'C']
    dfs = []
    for expert in experts:
        df = pd.read_csv(f'../data/annotations_2017_{expert}.csv')
        dfs.append(df)
        
    # Crear dataframe con la moda de los expertos
    combined_df = pd.DataFrame(columns= [i for i in range(1,80)])
    for row_a, row_b, row_c in zip(dfs[0].itertuples(), dfs[1].itertuples(), dfs[2].itertuples()):
        combined_row = [max(row_a[i], row_b[i], row_c[i], key = [row_a[i], row_b[i], row_c[i]].count) for i in range(1,80)] # moda de expertos
        combined_df.loc[len(combined_df)] = combined_row
    combined_df.to_csv('../data/labels.csv', index = False)
    return combined_df


def plot_labels():
    combined_df = pd.read_csv('../data/labels.csv')
    # Ver si exiten columnas donde hay solo 0s
    zeros = []
    for i in range(1,80):
        if not combined_df[i].any():
            zeros.append(i)
        
    print("Columnas con solo 0s:", zeros)

    # Graficar dataframe
    plt.figure(figsize = (20,10))
    plt.imshow(combined_df.T, aspect = 'auto', cmap = 'binary', interpolation = 'none')
    plt.xticks(np.arange(0, 60*24*60, 30*60), np.arange(0, 24, 0.5))
    plt.xlim(0, 60*60*4.5)
    for i in range(1,80): plt.axhline(i-0.5, color = 'black', linewidth = 0.5)
    plt.xlabel('Time (h)')
    plt.ylabel('Patient')
    plt.title('Annotations')
    ax = plt.gca()
    ax.set_facecolor('xkcd:light grey')
    plt.show()

def main():
    set_of_curves = set()
    for i in range(1, 2):#80
        print(f"Processing file {i}")
        proyect_path = os.path.abspath(os.getcwd())
        # raw_file = f"../data/eeg{i}.edf"
        raw_file = f"{proyect_path}/data/eeg{i}.edf"
        with warnings.catch_warnings(): # Ignore warnings
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(raw_file, preload = True, verbose = False)
        # Filtrar alta frecuencia
        notch_frequencies = [50, 100]
        raw.notch_filter(freqs = notch_frequencies, verbose = False)
        raw.filter(1, 50, fir_design = 'firwin', verbose = False)
        raw.resample(sfreq = 128)
        data = raw.get_data()
        data_curves = raw.ch_names
        data_curves = map(str.upper, data_curves)
        data_curves = [curve + "-REF" if not re.search("-REF", curve) else curve for curve in data_curves]
        data_curves = set(data_curves)
        set_of_curves = set_of_curves.union(data_curves)


    # freq_range = (0, 128)  # Puedes ajustar esto según tus necesidades.

    # # # Plotea el power spectrum.
    # raw.plot_psd(fmin=freq_range[0], fmax=freq_range[1], tmax=np.inf, show=True, average=True)

    # # Puedes personalizar el gráfico según tus preferencias.
    # plt.title('Power Spectrum of EEG')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power/Frequency (dB/Hz)')

    # # Muestra el gráfico.
    # plt.show()

    # # print(f"Total number of curves: {len(set_of_curves)}")
    # # for curve in set_of_curves:
    # #     print(curve)
        
    # # raw.info
    raw.plot(show_scrollbars=True)
    plt.show()

if __name__ == '__main__':
    main()