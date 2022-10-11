###################################################################################
#             PROYECTO: MODELO PARA DETERMINAR LA CATEGORÍA DE UN VINO
#                             CON MACHINE LEARNING
###################################################################################

#######################################
# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
#######################################

########################################
# IMPORTAMOS LAS LIBRERÍAS DE INTERÉS
########################################

import numpy as np
import pandas as pd
import random
from numpy.random import rand
import warnings
import seaborn as sns
warnings.filterwarnings('once')

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

seed=123
np.random.seed(seed) # fijamos la semilla
random.seed(seed)


##############################
# CARGAMOS LOS DATOS
##############################

data=pd.read_csv("data.csv",sep=",")
data.info()


##############################
# VISUALIZAMOS ALGUNOS CASOS
##############################

sns.set(font_scale=1.3)


cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']

variable_respuesta="quality"
features = list(data.columns)
features = list(set(features) - set([variable_respuesta]))

fig = plt.figure(figsize=(8,8))

for n, feature in enumerate(features):
    ax = plt.subplot(4, 3, n + 1)
    box = data[[feature, 'quality']].boxplot(by='quality',ax=ax,return_type='both',patch_artist = True, showfliers=False)

    for row_key, (ax,row) in box.iteritems():
        ax.set_xlabel('quality')
        ax.set_title(feature,fontweight="bold")
        for i,box in enumerate(row['boxes']):
            box.set_facecolor(cluster_colors[i])

fig.suptitle('Feature distributions per cluster', fontsize=16, y=1)
plt.tight_layout()
plt.show()



# PREPROCESAMIENTO

def preprocesamiento(data):
    # Escalamos los datos
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data.drop("quality", axis=1))
    # Since the fit_transform() strips the column headers
    # we add them after the transformation
    data_std = pd.DataFrame(data_std, columns=data.drop("quality", axis=1).columns)
    data_std = data_std.assign(quality=data['quality'])

    # Cálculamos el promedio de cada variable y por cada categoría
    X_mean = pd.concat([pd.DataFrame(data.mean().drop('quality'), columns=['mean']),
                        data.groupby('quality').mean().T], axis=1)

    # Cálculamos la desviación relativa (diferencia con el promedio dividida por el promedio)
    X_dev_rel = X_mean.apply(lambda x: round((x - x['mean']) / x['mean'], 2) * 100, axis=1)
    X_dev_rel.drop(columns=['mean'], inplace=True)
    X_mean.drop(columns=['mean'], inplace=True)

    # Cálculamos el promedio de la data estandarizada por variable y por categoría
    X_std_mean = pd.concat([pd.DataFrame(data_std.mean().drop('quality'), columns=['mean']),
                            data_std.groupby('quality').mean().T], axis=1)

    # Cálculamos la desviación de la data estandarizada
    X_std_dev_rel = X_std_mean.apply(lambda x: round((x - x['mean']) / x['mean'], 2) * 100, axis=1)
    X_std_dev_rel.drop(columns=['mean'], inplace=True)
    X_std_mean.drop(columns=['mean'], inplace=True)
    return X_dev_rel, X_std_mean


# Bar plots
import matplotlib.patches as mpatches


def cluster_comparison_bar(data,n_categorias, variable_respuesta, colors, title="Cluster results"):
    X_dev_rel, X_std_mean = preprocesamiento(data)
    features = set(list(data.columns)) - set([variable_respuesta])
    # set figure size
    fig = plt.figure(figsize=(16, 16), dpi=90)
    nrows=4
    ncols=3
    # interate through every feature
    for n, feature in enumerate(features):
        # create chart
        ax = plt.subplot(nrows,ncols, n + 1)
        X_dev_rel[X_dev_rel.index == feature].plot(kind='bar', ax=ax, title=feature,
                                                         color=colors[0:n_categorias],
                                                         legend=False
                                                         )
        plt.axhline(y=0)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)

    c_labels = X_dev_rel.columns.to_list()
    c_colors = colors[0:3]
    mpats = [mpatches.Patch(color=c, label=l) for c, l in list(zip(colors[0:n_categorias],
                                                                   X_dev_rel.columns.to_list()))]

    fig.legend(handles=mpats,
               ncol=ncols,
               loc='center right' ,
               fancybox=True,
               bbox_to_anchor=(0.98, 0.1)
               )
    axes = fig.get_axes()

    fig.suptitle(title, fontsize=18, y=1)
    fig.supylabel('Desviación porcentual en relación al promedio')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


cluster_comparison_bar(data,6,"quality", cluster_colors, title="Comparación de la desviación porcentual según categoría")



# Radar chart

class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0 / self.n)

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14, backgroundcolor="white",
                               zorder=999)  # Feature names
        self.ax.set_yticklabels([])

        for ax in self.axes[1:]:
            ax.xaxis.set_visible(False)
            ax.set_yticklabels([])
            ax.set_zorder(-99)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.spines['polar'].set_color('black')
            ax.spines['polar'].set_zorder(-99)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
        kw['label'] = '_noLabel'
        self.ax.fill(angle, values, *args, **kw)


fig = plt.figure(figsize=(6, 6))

no_features = len(features)
radar = Radar(fig, X_std_mean.index, data.quality.unique())

n=0
for k in data.quality.unique():
    cluster_data = X_std_mean[k].tolist()
    radar.plot(cluster_data, '-', lw=2, color=cluster_colors[n], alpha=0.7, label='cluster {}'.format(k))
    n=n+1

#radar.ax.legend()
radar.ax.legend(loc='center right',bbox_to_anchor=(0, 0.1))
radar.ax.set_title("Cluster characteristics: Feature means per cluster", size=22, pad=60)
plt.show()