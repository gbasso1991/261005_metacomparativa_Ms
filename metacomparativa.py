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
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
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
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
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
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return SAR, tau, Hc
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

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%%

#%% M21
nombres = ['M15','M16','M17','M18','M19','M20','M21']
temperatura = [235,235,237,215,250,226,243]
estufa = ['EN','EV','EN','EN','EN','EN','EN'] # Estufa Nueva/Vieja
autoclave = ['AuN','AuN','AuN','AuN','AuN','AuN','AuN'] #Autoclave Nuevo/Viejo
concentracion = [21.6, 16.5, 23.9, 22.5, 22, 20, 20] # g/L Magnetita

ciclos_M15 = glob("data_M15/*ciclo_promedio_H_M*")
resultados_M15 = glob("data_M15/*resultados*")
ciclos_M16 = glob("data_M16/*ciclo_promedio_H_M*")
resultados_M16 = glob("data_M16/*resultados*")
ciclos_M17 = glob("data_M17/*ciclo_promedio_H_M*")
resultados_M17 = glob("data_M17/*resultados*")
ciclos_M18 = glob("data_M18/*ciclo_promedio_H_M*")
resultados_M18 = glob("data_M18/*resultados*")
ciclos_M19 = glob("data_M19/*ciclo_promedio_H_M*")
resultados_M19 = glob("data_M19/*resultados*")
ciclos_M20 = glob("data_M20/*ciclo_promedio_H_M*")
resultados_M20 = glob("data_M20/*resultados*")
ciclos_M21 = glob("data_M21/*ciclo_promedio_H_M*")
resultados_M21 = glob("data_M21/*resultados*")
#%% Ordeno listas
for c in [ciclos_M15, ciclos_M16, ciclos_M17, ciclos_M18, ciclos_M19, ciclos_M20, ciclos_M21]:
    c.sort()

for r in [resultados_M15, resultados_M16, resultados_M17, resultados_M18, resultados_M19, resultados_M20, resultados_M21]:
    r.sort()    
#%% extraigo SAR, tau y Hc
SAR_M15, tau_M15, Hc_M15 = extraer_SAR_tau(resultados_M15)
res_M15=[]
SAR_M16, tau_M16, Hc_M16 = extraer_SAR_tau(resultados_M16)
res_M16=[]
SAR_M17, tau_M17, Hc_M17 = extraer_SAR_tau(resultados_M17)
res_M17=[]
SAR_M18, tau_M18, Hc_M18 = extraer_SAR_tau(resultados_M18)
res_M18=[]
SAR_M19, tau_M19, Hc_M19 = extraer_SAR_tau(resultados_M19)
res_M19=[]
SAR_M20, tau_M20, Hc_M20 = extraer_SAR_tau(resultados_M20)
res_M20=[]
SAR_M21, tau_M21, Hc_M21 = extraer_SAR_tau(resultados_M21)
res_M21=[]
#%% ploteo ciclos
fig00, axs =plt.subplots(7,3,figsize=(15,27),constrained_layout=True,sharey=True,sharex=True)
#M15
for i,e in enumerate(ciclos_M15):
    if '100dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('1',os.path.basename(e))
        axs[0,0].plot(H_M15/1000,M_M15,'-',label=f'{SAR_M15[i]:.2uS}')

    elif '125dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('2',os.path.basename(e))
        axs[0,1].plot(H_M15/1000,M_M15,'-',label=f'{SAR_M15[i]:.3uS}')

    elif '150dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('3',os.path.basename(e))
        axs[0,2].plot(H_M15/1000,M_M15,'-',label=f'{SAR_M15[i]:.3uS}')
#M16
for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('1',os.path.basename(e))
        axs[1,0].plot(H_M16/1000,M_M16,'-',label=f'{SAR_M16[i]:.3uS}')
    elif '125dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('2',os.path.basename(e))
        axs[1,1].plot(H_M16/1000,M_M16,'-',label=f'{SAR_M16[i]:.3uS}')

    elif '150dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('3',os.path.basename(e))
        axs[1,2].plot(H_M16/1000,M_M16,'-',label=f'{SAR_M16[i]:.3uS}')
#M17
for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('1',os.path.basename(e))
        axs[2,0].plot(H_M17/1000,M_M17,'-',label=f'{SAR_M17[i]:.2uS}')

    elif '125dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('2',os.path.basename(e))
        axs[2,1].plot(H_M17/1000,M_M17,'-',label=f'{SAR_M17[i]:.2uS}')

    elif '150dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('3',os.path.basename(e))
        axs[2,2].plot(H_M17/1000,M_M17,'-',label=f'{SAR_M17[i]:.2uS}')       
#M18
for i,e in enumerate(ciclos_M18):
    if '100dA' in e:
        _,_,_, H_M18,M_M18,_ = lector_ciclos(ciclos_M18[i])
        print('1',os.path.basename(e))
        axs[3,0].plot(H_M18/1000,M_M18,'-',label=f'{SAR_M18[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M18,M_M18,_ = lector_ciclos(ciclos_M18[i])
        print('2',os.path.basename(e))
        axs[3,1].plot(H_M18/1000,M_M18,'-',label=f'{SAR_M18[i]:.2uS}')

    elif '150dA' in e:
        _,_,_, H_M18,M_M18,_ = lector_ciclos(ciclos_M18[i])
        print('3',os.path.basename(e))
        axs[3,2].plot(H_M18/1000,M_M18,'-',label=f'{SAR_M18[i]:.2uS}')
#M19
for i,e in enumerate(ciclos_M19):
    if '100dA' in e:
        _,_,_, H_M19,M_M19,_ = lector_ciclos(ciclos_M19[i])
        print('1',os.path.basename(e))
        axs[4,0].plot(H_M19/1000,M_M19,'-',label=f'{SAR_M19[i]:.2uS}')

    elif '125dA' in e:
        _,_,_, H_M19,M_M19,_ = lector_ciclos(ciclos_M19[i])
        print('2',os.path.basename(e))
        axs[4,1].plot(H_M19/1000,M_M19,'-',label=f'{SAR_M19[i]:.2uS}')

    elif '150dA' in e:
        _,_,_, H_M19,M_M19,_ = lector_ciclos(ciclos_M19[i])
        print('3',os.path.basename(e))
        axs[4,2].plot(H_M19/1000,M_M19,'-',label=f'{SAR_M19[i]:.2uS}')
#M20
for i,e in enumerate(ciclos_M20):
    if '100dA' in e:
        _,_,_, H_M20,M_M20,_ = lector_ciclos(ciclos_M20[i])
        print('1',os.path.basename(e))
        axs[5,0].plot(H_M20/1000,M_M20,'-',label=f'{SAR_M20[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M20,M_M20,_ = lector_ciclos(ciclos_M20[i])
        print('2',os.path.basename(e))
        axs[5,1].plot(H_M20/1000,M_M20,'-',label=f'{SAR_M20[i]:.2uS}')

    elif '150dA' in e:
        _,_,_, H_M20,M_M20,_ = lector_ciclos(ciclos_M20[i])
        print('3',os.path.basename(e))
        axs[5,2].plot(H_M20/1000,M_M20,'-',label=f'{SAR_M20[i]:.3uS}')
#M21
for i,e in enumerate(ciclos_M21):
    if '100dA' in e:
        _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
        print('1',os.path.basename(e))
        axs[6,0].plot(H_M21/1000,M_M21,'-',label=f'{SAR_M21[i]:.2uS}')

    elif '125dA' in e:
        _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
        print('2',os.path.basename(e))
        axs[6,1].plot(H_M21/1000,M_M21,'-',label=f'{SAR_M21[i]:.2uS}')

    elif '150dA' in e:
        _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
        print('3',os.path.basename(e))
        axs[6,2].plot(H_M21/1000,M_M21,'-',label=f'{SAR_M21[i]:.3uS}')
        

for i,e in enumerate(axs.ravel()[::3]):
    e.set_title(nombres[i],loc='left')
    e.set_ylabel('M (A/m)')

for a in axs.ravel()[-3:]:
    a.set_xlabel('H (kA/m)')
for a in axs.ravel():
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio M15  M16  M17  M18  M19  M20  M21\n300 kHz & [38, 47, 57] kA/m')

#%%Ciclos promedio normalizados
fig01, axs =plt.subplots(7,3,figsize=(15,27),constrained_layout=True,sharey=True,sharex=True)
#M15
for i,e in enumerate(ciclos_M15):
    if '100dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('1',os.path.basename(e))
        axs[0,0].plot(H_M15/1000,M_M15/concentracion[0],'-',label=f'{SAR_M15[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('2',os.path.basename(e))
        axs[0,1].plot(H_M15/1000,M_M15/concentracion[0],'-',label=f'{SAR_M15[i]:.3uS}')
    elif '150dA' in e:
        _,_,_, H_M15,M_M15,_ = lector_ciclos(ciclos_M15[i])
        print('3',os.path.basename(e))
        axs[0,2].plot(H_M15/1000,M_M15/concentracion[0],'-',label=f'{SAR_M15[i]:.3uS}')    
#M16
for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('1',os.path.basename(e))
        axs[1,0].plot(H_M16/1000,M_M16/concentracion[1],'-',label=f'{SAR_M16[i]:.3uS}')
    elif '125dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('2',os.path.basename(e))
        axs[1,1].plot(H_M16/1000,M_M16/concentracion[1],'-',label=f'{SAR_M16[i]:.3uS}')
    elif '150dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('3',os.path.basename(e))
        axs[1,2].plot(H_M16/1000,M_M16/concentracion[1],'-',label=f'{SAR_M16[i]:.3uS}')
#M17
for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('1',os.path.basename(e))
        axs[2,0].plot(H_M17/1000,M_M17/concentracion[2],'-',label=f'{SAR_M17[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('2',os.path.basename(e))
        axs[2,1].plot(H_M17/1000,M_M17/concentracion[2],'-',label=f'{SAR_M17[i]:.2uS}')
    elif '150dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('3',os.path.basename(e))
        axs[2,2].plot(H_M17/1000,M_M17/concentracion[2],'-',label=f'{SAR_M17[i]:.3uS}')      
#M18
for i,e in enumerate(ciclos_M18):
    if '100dA' in e:
        _,_,_, H_M18,M_M18,_ = lector_ciclos(ciclos_M18[i])
        print('1',os.path.basename(e))
        axs[3,0].plot(H_M18/1000,M_M18/concentracion[3],'-',label=f'{SAR_M18[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M18,M_M18,_ = lector_ciclos(ciclos_M18[i])
        print('2',os.path.basename(e))
        axs[3,1].plot(H_M18/1000,M_M18/concentracion[3],'-',label=f'{SAR_M18[i]:.2uS}')
    elif '150dA' in e:
        _,_,_, H_M18,M_M18,_ = lector_ciclos(ciclos_M18[i])
        print('3',os.path.basename(e))
        axs[3,2].plot(H_M18/1000,M_M18/concentracion[3],'-',label=f'{SAR_M18[i]:.2uS}')
#M19
for i,e in enumerate(ciclos_M19):
    if '100dA' in e:
        _,_,_, H_M19,M_M19,_ = lector_ciclos(ciclos_M19[i])
        print('1',os.path.basename(e))
        axs[4,0].plot(H_M19/1000,M_M19/concentracion[4],'-',label=f'{SAR_M19[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M19,M_M19,_ = lector_ciclos(ciclos_M19[i])
        print('2',os.path.basename(e))
        axs[4,1].plot(H_M19/1000,M_M19/concentracion[4],'-',label=f'{SAR_M19[i]:.2uS}')
    elif '150dA' in e:
        _,_,_, H_M19,M_M19,_ = lector_ciclos(ciclos_M19[i])
        print('3',os.path.basename(e))
        axs[4,2].plot(H_M19/1000,M_M19/concentracion[4],'-',label=f'{SAR_M19[i]:.2uS}')   
#M20
for i,e in enumerate(ciclos_M20):
    if '100dA' in e:
        _,_,_, H_M20,M_M20,_ = lector_ciclos(ciclos_M20[i])
        print('1',os.path.basename(e))
        axs[5,0].plot(H_M20/1000,M_M20/concentracion[5],'-',label=f'{SAR_M20[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M20,M_M20,_ = lector_ciclos(ciclos_M20[i])
        print('2',os.path.basename(e))
        axs[5,1].plot(H_M20/1000,M_M20/concentracion[5],'-',label=f'{SAR_M20[i]:.2uS}')
    elif '150dA' in e:
        _,_,_, H_M20,M_M20,_ = lector_ciclos(ciclos_M20[i])
        print('3',os.path.basename(e))
        axs[5,2].plot(H_M20/1000,M_M20/concentracion[5],'-',label=f'{SAR_M20[i]:.3uS}')
#M21
for i,e in enumerate(ciclos_M21):
    if '100dA' in e:
        _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
        print('1',os.path.basename(e))
        axs[6,0].plot(H_M21/1000,M_M21/concentracion[6],'-',label=f'{SAR_M21[i]:.2uS}')
    elif '125dA' in e:
        _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
        print('2',os.path.basename(e))
        axs[6,1].plot(H_M21/1000,M_M21/concentracion[6],'-',label=f'{SAR_M21[i]:.2uS}')
    elif '150dA' in e:
        _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
        print('3',os.path.basename(e))
        axs[6,2].plot(H_M21/1000,M_M21/concentracion[6],'-',label=f'{SAR_M21[i]:.3uS}')
        
for i,e in enumerate(axs.ravel()[::3]):
    e.set_title(nombres[i],loc='left')
    e.set_ylabel('M/[NPM] (Am²/kg)')

for a in axs.ravel()[-3:]:
    a.set_xlabel('H (kA/m)')
for i,a in enumerate(axs.ravel()):
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio normalizados por concentración\nM15  M16  M17  M18  M19  M20  M21\n300 kHz & [38, 47, 57] kA/m')

#%%
fig00.savefig('00_comparativa_ciclos.png',dpi=300)
fig01.savefig('01_comparativa_ciclos_normalizados.png',dpi=300)

#%%% Ciclos todos  Normalizado por concentracion
# fig01, axs =plt.subplots(1,1,figsize=(9,7),constrained_layout=True,sharey=True,sharex=True)
# axs.set_ylabel('M (A/m)')
# lines=['-','--','-.',':']*3
# ejes=[-57 , -47, -38,0, 38, 47, 57]
# str_ejes=[str(e) for e in ejes] 




# for i,e in enumerate(ciclos_M21):
#     if '100dA' in e:
#         _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
#         print('1',os.path.basename(e))
#         axs.plot(H_M21/1000,M_M21,c='C0',ls=lines[i],label=f'{SAR_M21[i]:.3uS}')
#     elif '125dA' in e:
#         _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
#         print('2',os.path.basename(e))
#         axs.plot(H_M21/1000,M_M21,c='C1',ls=lines[i],label=f'{SAR_M21[i]:.3uS}')
#     elif '150dA' in e:
#         _,_,_, H_M21,M_M21,_ = lector_ciclos(ciclos_M21[i])
#         print('3',os.path.basename(e))
#         axs.plot(H_M21/1000,M_M21,c='C2',ls=lines[i],label=f'{SAR_M21[i]:.3uS}')

# axs.set_xticks(ejes)
# axs.set_xticklabels(str_ejes)
# axs.grid()
# axs.set_xlabel('H (kA/m)')
# axs.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)',ncol=1)
# plt.suptitle(f'Ciclos promedio {nombre_M21} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M21:.1f} g/L')


#%%
res_M15 = []

print('Resultados M15', '='*80,'\n')
for r in resultados_M15:
    res_M15.append(ResultadosESAR(os.path.dirname(r)))
rates_M15 = []
#%%
print('Resultados M16', '='*80,'\n')
for r in resultados_M16:
    res_M16.append(ResultadosESAR(os.path.dirname(r)))
rates_M16 = []
print('Resultados M17', '='*80,'\n')
for r in resultados_M17:
    res_M17.append(ResultadosESAR(os.path.dirname(r)))
rates_M17 = []
print('Resultados M18', '='*80,'\n')
for r in resultados_M18:
    res_M18.append(ResultadosESAR(os.path.dirname(r)))
rates_M18 = []
print('Resultados M19', '='*80,'\n')
for r in resultados_M19:
    res_M19.append(ResultadosESAR(os.path.dirname(r)))
rates_M19 = []
print('Resultados M20', '='*80,'\n')
for r in resultados_M20:
    res_M20.append(ResultadosESAR(os.path.dirname(r)))
rates_M20 = []
print('Resultados M21', '='*80,'\n')
for r in resultados_M21:
    res_M21.append(ResultadosESAR(os.path.dirname(r)))
rates_M21 = []

#%% Templogs
fig02, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')


for i,r in enumerate(res_M21):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M21.append(rate)
    print(i,f'WRate = {rate:.2f} °C/s')
    if '100dA' in r.directorio:
        axs[0].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')
    elif '125dA' in r.directorio:
        axs[1].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')
    elif '150dA' in r.directorio:
        axs[2].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')

axs[0].set_ylabel('T (°C)')
for a in axs:
    a.grid()
    a.set_xlabel('t (s)')
    a.legend(loc='best',frameon=True,shadow=True,title='Warming Rate (°C/s)')
plt.suptitle(f'Templogs {nombre_M21} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M21:.1f} g/L')

################################################################################################################################

#%% ploteo comparativo de errorbars de ESAR
categorias = ['38 kA/m', '47 kA/m', '57 kA/m']
x = np.arange(len(categorias))

fig03, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)

sep = 0.25

for i,s in enumerate(SAR_M21[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_M21[3:6]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for j,s in enumerate(SAR_M21[6:]):
    ax.bar(j*sep + 7*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')

for i,s in enumerate(tau_M21[:3]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_M21[3:6]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for j,s in enumerate(tau_M21[6:]):
    ax2.bar(j*sep + 7*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')

for i,s in enumerate(Hc_M21[:3]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_M21[3:6]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for j,s in enumerate(Hc_M21[6:]):
    ax3.bar(j*sep + 7*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')


for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
    

ax.set_title('ESAR',loc='left')
ax2.set_title('tau',loc='left')
ax3.set_title('Hc',loc='left')
ax.set_ylabel('ESAR (W/g)')
ax2.set_ylabel('tau (ns)')
ax3.set_ylabel('Hc (kA/m)')
plt.suptitle(f'Parametros {nombre_M21} \n300 kHz - [38 , 47, 57] kA/m')
plt.show()

#%% Salvo figuras
fig00.savefig('00_ciclos_M21.png',dpi=300)
fig01.savefig('01_ciclos_M21_norm.png',dpi=300)
fig02.savefig('02_templogs_M21.png',dpi=300)
fig03.savefig('03_comparativa_M21_ESAR_Tau_Hc.png',dpi=300)
# %%
