#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	ESAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    ESAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, ESAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentraciong/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float) #kA/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer ESAR, tau y Hc de resultados
def extraer_ESAR_tau(resultados):
    ESAR, tau, Hc = [], [], []
    time , temperatura = [], []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        ESAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return ESAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.aEsarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% funcion Warming rate promedio por campo
def warming_rates_promedio(resultados):
    
    rates = []

    # warming rate de cada una de las 9 mediciones
    for r in resultados:
        dt = r.time[-1] - r.time[0]
        dT = r.temperatura[-1] - r.temperatura[0]
        rates.append(dT / dt)

    # promedio de cada grupo de 3
    rates_promedio = []

    for i in range(0, 9, 3):
        grupo = rates[i:i+3]
        
        promedio = np.mean(grupo)
        std = np.std(grupo, ddof=1)
        
        rates_promedio.append(ufloat(promedio, std))

    return rates_promedio
#%% Importo ciclos y resultados  ordenado por temperatura
nombres = ['M18','M20','M15','M16','M17','M21','M19']
temperatura = [215,226,235,235,237,243,250]
concentracion = [22.5,20,21.6,16.5,23.9,20,22] # g/L Magnetita
nombres = ['M18', 'M20', 'M15', 'M16', 'M17', 'M21', 'M19']
ciclos_M18 = glob("data_M18/*ciclo_promedio_H_M*")
resultados_M18 = glob("data_M18/*resultados*")
ciclos_M20 = glob("data_M20/*ciclo_promedio_H_M*")
resultados_M20 = glob("data_M20/*resultados*")
ciclos_M15 = glob("data_M15/*ciclo_promedio_H_M*")
resultados_M15 = glob("data_M15/*resultados*")
ciclos_M16 = glob("data_M16/*ciclo_promedio_H_M*")
resultados_M16 = glob("data_M16/*resultados*")
ciclos_M17 = glob("data_M17/*ciclo_promedio_H_M*")
resultados_M17 = glob("data_M17/*resultados*")
ciclos_M21 = glob("data_M21/*ciclo_promedio_H_M*")
resultados_M21 = glob("data_M21/*resultados*")
ciclos_M19 = glob("data_M19/*ciclo_promedio_H_M*")
resultados_M19 = glob("data_M19/*resultados*")
#%% Ordeno listas
for c in [ciclos_M18, ciclos_M20, ciclos_M15, ciclos_M16, ciclos_M17, ciclos_M19, ciclos_M21]:
    c.sort()

for r in [resultados_M18, resultados_M20, resultados_M15, resultados_M16, resultados_M17, resultados_M19, resultados_M21]:
    r.sort()    
#%% extraigo ESAR, tau y Hc
ESAR_M18, tau_M18, Hc_M18 = extraer_ESAR_tau(resultados_M18)
res_M18=[]
ESAR_M20, tau_M20, Hc_M20 = extraer_ESAR_tau(resultados_M20)
res_M20=[]
ESAR_M15, tau_M15, Hc_M15 = extraer_ESAR_tau(resultados_M15)
res_M15=[]
ESAR_M16, tau_M16, Hc_M16 = extraer_ESAR_tau(resultados_M16)
res_M16=[]
ESAR_M17, tau_M17, Hc_M17 = extraer_ESAR_tau(resultados_M17)
res_M17=[]
ESAR_M21, tau_M21, Hc_M21 = extraer_ESAR_tau(resultados_M21)
res_M21=[]
ESAR_M19, tau_M19, Hc_M19 = extraer_ESAR_tau(resultados_M19)
res_M19=[]
#%% ploteo ciclos
fig00,axs=plt.subplots(7,3,figsize=(15,27),constrained_layout=True,sharey=True,sharex=True)
#M18
for i,e in enumerate(ciclos_M18):
    if '100dA' in e:
        _,_,_,H_M18,M_M18,_=lector_ciclos(ciclos_M18[i])
        axs[0,0].plot(H_M18,M_M18,'-',color='C0',label=f'{ESAR_M18[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M18,M_M18,_=lector_ciclos(ciclos_M18[i])
        axs[0,1].plot(H_M18,M_M18,'-',color='C1',label=f'{ESAR_M18[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M18,M_M18,_=lector_ciclos(ciclos_M18[i])
        axs[0,2].plot(H_M18,M_M18,'-',color='C2',label=f'{ESAR_M18[i]:.2uS}')
#M20
for i,e in enumerate(ciclos_M20):
    if '100dA' in e:
        _,_,_,H_M20,M_M20,_=lector_ciclos(ciclos_M20[i])
        axs[1,0].plot(H_M20,M_M20,'-',color='C0',label=f'{ESAR_M20[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M20,M_M20,_=lector_ciclos(ciclos_M20[i])
        axs[1,1].plot(H_M20,M_M20,'-',color='C1',label=f'{ESAR_M20[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M20,M_M20,_=lector_ciclos(ciclos_M20[i])
        axs[1,2].plot(H_M20,M_M20,'-',color='C2',label=f'{ESAR_M20[i]:.3uS}')
#M15
for i,e in enumerate(ciclos_M15):
    if '100dA' in e:
        _,_,_,H_M15,M_M15,_=lector_ciclos(ciclos_M15[i])
        axs[2,0].plot(H_M15,M_M15,'-',color='C0',label=f'{ESAR_M15[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M15,M_M15,_=lector_ciclos(ciclos_M15[i])
        axs[2,1].plot(H_M15,M_M15,'-',color='C1',label=f'{ESAR_M15[i]:.3uS}')
    elif '150dA' in e:
        _,_,_,H_M15,M_M15,_=lector_ciclos(ciclos_M15[i])
        axs[2,2].plot(H_M15,M_M15,'-',color='C2',label=f'{ESAR_M15[i]:.3uS}')
#M16
for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_,H_M16,M_M16,_=lector_ciclos(ciclos_M16[i])
        axs[3,0].plot(H_M16,M_M16,'-',color='C0',label=f'{ESAR_M16[i]:.3uS}')
    elif '125dA' in e:
        _,_,_,H_M16,M_M16,_=lector_ciclos(ciclos_M16[i])
        axs[3,1].plot(H_M16,M_M16,'-',color='C1',label=f'{ESAR_M16[i]:.3uS}')
    elif '150dA' in e:
        _,_,_,H_M16,M_M16,_=lector_ciclos(ciclos_M16[i])
        axs[3,2].plot(H_M16,M_M16,'-',color='C2',label=f'{ESAR_M16[i]:.3uS}')
#M17
for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_,H_M17,M_M17,_=lector_ciclos(ciclos_M17[i])
        axs[4,0].plot(H_M17,M_M17,'-',color='C0',label=f'{ESAR_M17[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M17,M_M17,_=lector_ciclos(ciclos_M17[i])
        axs[4,1].plot(H_M17,M_M17,'-',color='C1',label=f'{ESAR_M17[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M17,M_M17,_=lector_ciclos(ciclos_M17[i])
        axs[4,2].plot(H_M17,M_M17,'-',color='C2',label=f'{ESAR_M17[i]:.2uS}')
#M21
for i,e in enumerate(ciclos_M21):
    if '100dA' in e:
        _,_,_,H_M21,M_M21,_=lector_ciclos(ciclos_M21[i])
        axs[5,0].plot(H_M21,M_M21,'-',color='C0',label=f'{ESAR_M21[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M21,M_M21,_=lector_ciclos(ciclos_M21[i])
        axs[5,1].plot(H_M21,M_M21,'-',color='C1',label=f'{ESAR_M21[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M21,M_M21,_=lector_ciclos(ciclos_M21[i])
        axs[5,2].plot(H_M21,M_M21,'-',color='C2',label=f'{ESAR_M21[i]:.3uS}')
#M19
for i,e in enumerate(ciclos_M19):
    if '100dA' in e:
        _,_,_,H_M19,M_M19,_=lector_ciclos(ciclos_M19[i])
        axs[6,0].plot(H_M19,M_M19,'-',color='C0',label=f'{ESAR_M19[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M19,M_M19,_=lector_ciclos(ciclos_M19[i])
        axs[6,1].plot(H_M19,M_M19,'-',color='C1',label=f'{ESAR_M19[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M19,M_M19,_=lector_ciclos(ciclos_M19[i])
        axs[6,2].plot(H_M19,M_M19,'-',color='C2',label=f'{ESAR_M19[i]:.2uS}')

for i,e in enumerate(axs.ravel()[::3]):
    e.set_title(f'{nombres[i]} — {temperatura[i]} °C — {concentracion[i]} g/L',loc='left')
    e.set_ylabel('M (A/m)')
for a in axs.ravel()[-3:]:
    a.set_xlabel('H (kA/m)')
for a in axs.ravel():
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle('Ciclos promedio\n300 kHz — [38, 47, 57] kA/m',fontsize=14)

#%% #%% Ciclos promedio normalizados
fig01,axs=plt.subplots(7,3,figsize=(15,27),constrained_layout=True,sharey=True,sharex=True)
#M18
for i,e in enumerate(ciclos_M18):
    if '100dA' in e:
        _,_,_,H_M18,M_M18,_=lector_ciclos(ciclos_M18[i])
        axs[0,0].plot(H_M18,M_M18/concentracion[3],'-',color='C0',label=f'{ESAR_M18[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M18,M_M18,_=lector_ciclos(ciclos_M18[i])
        axs[0,1].plot(H_M18,M_M18/concentracion[3],'-',color='C1',label=f'{ESAR_M18[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M18,M_M18,_=lector_ciclos(ciclos_M18[i])
        axs[0,2].plot(H_M18,M_M18/concentracion[3],'-',color='C2',label=f'{ESAR_M18[i]:.2uS}')
#M20
for i,e in enumerate(ciclos_M20):
    if '100dA' in e:
        _,_,_,H_M20,M_M20,_=lector_ciclos(ciclos_M20[i])
        axs[1,0].plot(H_M20,M_M20/concentracion[5],'-',color='C0',label=f'{ESAR_M20[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M20,M_M20,_=lector_ciclos(ciclos_M20[i])
        axs[1,1].plot(H_M20,M_M20/concentracion[5],'-',color='C1',label=f'{ESAR_M20[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M20,M_M20,_=lector_ciclos(ciclos_M20[i])
        axs[1,2].plot(H_M20,M_M20/concentracion[5],'-',color='C2',label=f'{ESAR_M20[i]:.3uS}')
#M15
for i,e in enumerate(ciclos_M15):
    if '100dA' in e:
        _,_,_,H_M15,M_M15,_=lector_ciclos(ciclos_M15[i])
        axs[2,0].plot(H_M15,M_M15/concentracion[0],'-',color='C0',label=f'{ESAR_M15[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M15,M_M15,_=lector_ciclos(ciclos_M15[i])
        axs[2,1].plot(H_M15,M_M15/concentracion[0],'-',color='C1',label=f'{ESAR_M15[i]:.3uS}')
    elif '150dA' in e:
        _,_,_,H_M15,M_M15,_=lector_ciclos(ciclos_M15[i])
        axs[2,2].plot(H_M15,M_M15/concentracion[0],'-',color='C2',label=f'{ESAR_M15[i]:.3uS}')
#M16
for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_,H_M16,M_M16,_=lector_ciclos(ciclos_M16[i])
        axs[3,0].plot(H_M16,M_M16/concentracion[1],'-',color='C0',label=f'{ESAR_M16[i]:.3uS}')
    elif '125dA' in e:
        _,_,_,H_M16,M_M16,_=lector_ciclos(ciclos_M16[i])
        axs[3,1].plot(H_M16,M_M16/concentracion[1],'-',color='C1',label=f'{ESAR_M16[i]:.3uS}')
    elif '150dA' in e:
        _,_,_,H_M16,M_M16,_=lector_ciclos(ciclos_M16[i])
        axs[3,2].plot(H_M16,M_M16/concentracion[1],'-',color='C2',label=f'{ESAR_M16[i]:.3uS}')
#M17
for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_,H_M17,M_M17,_=lector_ciclos(ciclos_M17[i])
        axs[4,0].plot(H_M17,M_M17/concentracion[2],'-',color='C0',label=f'{ESAR_M17[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M17,M_M17,_=lector_ciclos(ciclos_M17[i])
        axs[4,1].plot(H_M17,M_M17/concentracion[2],'-',color='C1',label=f'{ESAR_M17[i]:.3uS}')
    elif '150dA' in e:
        _,_,_,H_M17,M_M17,_=lector_ciclos(ciclos_M17[i])
        axs[4,2].plot(H_M17,M_M17/concentracion[2],'-',color='C2',label=f'{ESAR_M17[i]:.3uS}')
#M21
for i,e in enumerate(ciclos_M21):
    if '100dA' in e:
        _,_,_,H_M21,M_M21,_=lector_ciclos(ciclos_M21[i])
        axs[5,0].plot(H_M21,M_M21/concentracion[6],'-',color='C0',label=f'{ESAR_M21[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M21,M_M21,_=lector_ciclos(ciclos_M21[i])
        axs[5,1].plot(H_M21,M_M21/concentracion[6],'-',color='C1',label=f'{ESAR_M21[i]:.2uS}')
    elif '150dA' in e:
        _,_,_,H_M21,M_M21,_=lector_ciclos(ciclos_M21[i])
        axs[5,2].plot(H_M21,M_M21/concentracion[6],'-',color='C2',label=f'{ESAR_M21[i]:.3uS}')
#M19
for i,e in enumerate(ciclos_M19):
    if '100dA' in e:
        _,_,_,H_M19,M_M19,_=lector_ciclos(ciclos_M19[i])
        axs[6,0].plot(H_M19,M_M19/concentracion[4],'-',color='C0',label=f'{ESAR_M19[i]:.2uS}')
    elif '125dA' in e:
        _,_,_,H_M19,M_M19,_=lector_ciclos(ciclos_M19[i])
        axs[6,1].plot(H_M19,M_M19/concentracion[4],'-',color='C1',label=f'{ESAR_M19[i]:.3uS}')
    elif '150dA' in e:
        _,_,_,H_M19,M_M19,_=lector_ciclos(ciclos_M19[i])
        axs[6,2].plot(H_M19,M_M19/concentracion[4],'-',color='C2',label=f'{ESAR_M19[i]:.3uS}')

for i,e in enumerate(axs.ravel()[::3]):
    e.set_title(f'{nombres[i]} — {temperatura[i]} °C — {concentracion[i]} g/L',loc='left')
    e.set_ylabel('M/[NPM] (Am²/kg)')
for a in axs.ravel()[-3:]:
    a.set_xlabel('H (kA/m)')
for a in axs.ravel():
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle('Ciclos promedio normalizados por concentracion\n300 kHz — [38, 47, 57] kA/m',fontsize=14)
plt.show()

#%% Extraigo resultados 
res_M15 = []
print('Resultados M15', '='*80,'\n')
for r in resultados_M15:
    res_M15.append(ResultadosESAR(r))

print('Resultados M16', '='*80,'\n')
for r in resultados_M16:
    res_M16.append(ResultadosESAR(r))

print('Resultados M17', '='*80,'\n')
for r in resultados_M17:
    res_M17.append(ResultadosESAR(r))

res_M18 = []
for r in resultados_M18:
    res_M18.append(ResultadosESAR(r))

print('Resultados M19', '='*80,'\n')
for r in resultados_M19:
    res_M19.append(ResultadosESAR(r))

print('Resultados M20', '='*80,'\n')
for r in resultados_M20:
    res_M20.append(ResultadosESAR(r))

print('Resultados M21', '='*80,'\n')
for r in resultados_M21:
    res_M21.append(ResultadosESAR(r))

WR_M18 = warming_rates_promedio(res_M18)
WR_M20 = warming_rates_promedio(res_M20)
WR_M15 = warming_rates_promedio(res_M15)
WR_M16 = warming_rates_promedio(res_M16)
WR_M17 = warming_rates_promedio(res_M17)
WR_M21 = warming_rates_promedio(res_M21)
WR_M19 = warming_rates_promedio(res_M19)

#%% Templogs — todas las muestras

nombres = ['M18', 'M20', 'M15', 'M16', 'M17', 'M21', 'M19']
temperatura = [215, 226, 235, 235, 237, 243, 250]
concentracion = [22.5, 20, 21.6, 16.5, 23.9, 20, 22]
campos = ['38 kA/m', '47 kA/m', '57 kA/m']
colores = ['C0', 'C1', 'C2','C0', 'C1', 'C2','C0', 'C1', 'C2']
resultados = [res_M18,res_M20,res_M15,res_M16,res_M17,res_M21,res_M19]

warming_rates = [WR_M18,WR_M20,WR_M15,WR_M16,WR_M17,WR_M21,WR_M19]

fig02, axs = plt.subplots(1, 7,figsize=(27, 5),constrained_layout=True,sharex=True,sharey=True)
for j, (nombre, res, WR) in enumerate(zip(nombres, resultados, warming_rates)):
    ax = axs[j]
    for i, r in enumerate(res):
        campo = i // 3
        ax.plot(r.time,r.temperatura,'.-',color=colores[campo],label=f'{WR[campo]:.1uS}' if i % 3 == 0 else None)

    ax.set_title(f'{nombre} — {temperatura[j]} °C — {concentracion[j]} g/L',loc='left')

    ax.set_xlabel('t (s)')
    ax.grid()

    ax.legend(title='Warming rate (°C/s)',loc='best',frameon=True,shadow=True)

axs[0].set_ylabel('T (°C)')
plt.suptitle('Templogs\n300 kHz — [38, 47, 57] kA/m',fontsize=16)
plt.show()
#%% ploteo comparativo de errorbars de ESAR tau y Hc

nombres = ['M18','M20','M15','M16','M17','M21','M19']
muestras = ['M18','M20','M15','M16','M17','M21','M19']

temperatura = [215,226,235,235,237,243,250]
concentracion = [22.5,20,21.6,16.5,23.9,20,22]
categorias = ['38 kA/m', '47 kA/m', '57 kA/m']

ESAR = [ESAR_M18, ESAR_M20, ESAR_M15, ESAR_M16, ESAR_M17, ESAR_M21, ESAR_M19]
tau = [tau_M18, tau_M20, tau_M15, tau_M16, tau_M17, tau_M21, tau_M19]
Hc = [Hc_M18, Hc_M20, Hc_M15, Hc_M16, Hc_M17, Hc_M21, Hc_M19]

datos = [ESAR, tau, Hc]

x = np.arange(len(categorias))
sep = 0.25

fig03, axs = plt.subplots(3, 7,figsize=(24, 8),sharey='row',sharex='col',constrained_layout=True)

axs[0,0].set_ylabel('ESAR (W/g)')
axs[1,0].set_ylabel(r'$tau$ (ns)')
axs[2,0].set_ylabel(r'$H_c$ (kA/m)')

for j, muestra in enumerate(muestras):

    axs[0,j].set_title(muestra + f' - {temperatura[j]} °C - {concentracion[j]} g/L',loc='left')
for fila, datos_fila in enumerate(datos):
    for col, muestra in enumerate(datos_fila):
        ax = axs[fila, col]
        for i, s in enumerate(muestra[:3]): # 38 kA/m
            ax.bar(i*sep - sep,s.n,yerr=s.s,width=0.2,capsize=5,color='C0')

        for i, s in enumerate(muestra[3:6]):# 47 kA/m
            ax.bar(i*sep + 3*sep,s.n,yerr=s.s,width=0.2,capsize=5,color='C1')

        for i, s in enumerate(muestra[6:9]):# 57 kA/m
            ax.bar(i*sep + 7*sep,s.n,yerr=s.s,width=0.2,capsize=5,color='C2')

for a in axs.ravel():
    a.grid(axis='y', alpha=0.7)

for ax in axs[2,:]:
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)

plt.suptitle('Comparativa ESAR - $\ttau$ - $H_c$ \n300 kHz - [38, 47, 57] kA/m',fontsize=16)

plt.show()


#%% Salvo figuras
fig00.savefig('00_metacomparativa_ciclos.png',dpi=300)
fig01.savefig('01_metacomparativa_ciclos_normalizados.png',dpi=300)
fig02.savefig('02_metacomparativa_templogs.png',dpi=300)
fig03.savefig('03_metacomparativa_ESAR_Tau_Hc.png',dpi=300)
# %%
