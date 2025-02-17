import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
from scipy.signal import butter, filtfilt

url_acc = ""
url_loc = ""
df_acc = pd.read_csv('./Linear Accelerometer.csv')
df_loc = pd.read_csv('./Location.csv')

st.title('Iltakävely')



#SUODATETTU ASKELDATA

def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

time = df_acc['Time (s)']

T = df_acc['Time (s)'].max() #Koko datan pituus
n = len(df_acc['Time (s)']) #Datapisteiden lukumäärä
fs = n/T #Näytteenottotaajuus
data = df_acc['Z (m/s^2)']

nyq = fs/2 #Nyquist-taajuus
order = 3 #Suodattimen aste
cutoff = 1/0.6 #cut-off-taajuus

filtered_signal = butter_lowpass_filter(data, cutoff, fs, nyq, order)

#Askelten määrä suodatetusta signaalista
jaksot = 0
for i in range(n-1):
    if filtered_signal[i]/filtered_signal[i+1] < 0:
        jaksot += 1

total_steps_from_filtered_data = jaksot/2 #Yhteenlasketut ylitykset jaettuna kahdella, koska ylitys tapahtuu molempiin suuntiin




# TEHOSPEKTRI

time = df_acc['Time (s)']
data = df_acc['Z (m/s^2)']

N = len(time)   #Datapisteiden lukumäärä
T = time.max() - time.min() #Kokonaisaika
dt = T / N

fourier = np.fft.fft(data, N)
psd = fourier*np.conj(fourier)/N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, np.floor(N/2), dtype='int')



# ASKELEET TEHOSPEKTRISTÄ

f_max = freq[L][psd[L] == np.max(psd[L])][0]
total_steps_from_fourier_analysis = T * f_max


#GPS

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

#Lasketaan nopeus ja kuljettu matka

df_loc['dist'] = np.zeros(len(df_loc)) #Etäisyys pisteiden välillä
df_loc['time_diff'] = np.zeros(len(df_loc)) #Aika pisteiden välillä

for i in range(len(df_loc)-1):
    #Distance between measurement point
    df_loc.loc[i+1, 'dist'] = haversine(df_loc['Longitude (°)'][i], df_loc['Latitude (°)'][i], df_loc['Longitude (°)'][i+1], df_loc['Latitude (°)'][i+1])*1000
    #Time between measurement points
    df_loc.loc[i+1, 'time_diff'] = df_loc['Time (s)'][i+1] - df_loc['Time (s)'][i]
    df_loc['velocity'] = df_loc['dist']/df_loc['time_diff']
    df_loc['tot_dist'] = np.cumsum(df_loc['dist'])

average_speed = df_loc['velocity'].mean()       #m/s
travelled_distance = df_loc['tot_dist'].max()   #metriä

start_lat = df_loc['Latitude (°)'].mean()
start_lon = df_loc['Longitude (°)'].mean()

map = folium.Map(location = [start_lat, start_lon], zoom_start = 14)



# TULOSTETAAN DATA STREAMLIT KOOSTEESEEN

#Tulostetaan laskettu data
st.title("Laskettu Data")
st.write("Askeleet suodatetusta datasta: ", total_steps_from_filtered_data, "askelta")
st.write("Askeleet Fourier-analyysilla: ", total_steps_from_fourier_analysis, "askelta")
st.write("Keskinopeus: ", f"{average_speed:.2f}", 'm/s' )
st.write("Kuljettu matka: ", f"{travelled_distance / 1000:.2f}", 'km')
average_steps = (total_steps_from_fourier_analysis + total_steps_from_filtered_data) / 2
average_step_length = travelled_distance / average_steps
st.write("Keskimääräinen askeleen pituus: ", f"{average_step_length:.2f}", "m")


#Suodatettu askeldata
st.title("Suodatettu Kiihtyvyysdata Z-akseli")
df_filtered = pd.DataFrame({
    "Aika (s)": time,
    "Suodatettu Kiihtyvyys": filtered_signal
}).set_index("Aika (s)")
st.line_chart(df_filtered, x_label="Aika (s)", y_label="Kiihtyvyys (m/s^2)")


#Kävelytaajuuden tehospektri
st.title("Kävelytaajuuden Tehospektri")

chart_data = pd.DataFrame(np.transpose(np.array([freq[L],psd[L].real])), columns=["freq", "psd"])
st.line_chart(chart_data, x = 'freq', y = 'psd' , y_label = 'Teho',x_label = 'Taajuus [Hz]')


#Piirretään kartta
st.title("Kävelty Reitti")

folium.PolyLine(df_loc[['Latitude (°)', 'Longitude (°)']], color = 'red', weight = 2.5, opacity = 1).add_to(map)
st_map = st_folium(map, width=900, height=650)