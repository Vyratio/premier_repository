#!/usr/bin/env python
# coding: utf-8

# In[585]:


import matplotlib.pyplot as plt
import datetime
import numpy as np
import plotly.express as px
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import f_oneway,chi2_contingency


# In[586]:


import sqlite3
import pandas as pd
conn_temperature = sqlite3.connect("temperature.sqlite3")
query_temperature = "SELECT * FROM Temperature"
data_temperature = pd.read_sql_query(query_temperature, conn_temperature)
data_temperature


# In[587]:


conn_objet_trouve = sqlite3.connect("objet_trouve.sqlite3")
query_objet_trouve = "SELECT * FROM Objet_trouve"
data_objet_trouve = pd.read_sql_query(query_objet_trouve, conn_objet_trouve)
data_objet_trouve


# In[588]:


#Voir calendar et group by pour créer par semaine
df_semaine =data_objet_trouve.groupby(by=["date"]).sum()
df_semaine


# In[589]:


conn_villes_gares = sqlite3.connect("villes_gares.sqlite3")
query_villes_gares = "SELECT * FROM Ville_Gare"
data_villes_gares = pd.read_sql_query(query_villes_gares, conn_villes_gares)
data_villes_gares


# In[590]:


list_departement = []
list_departement = data_villes_gares["département"].unique()
list(list_departement)


# In[591]:


list_departement = list(list_departement)


# Question n°1

# In[592]:


data_objet_trouve_modifie = data_objet_trouve
#df['A'] = df['A'].apply(lambda x: 0 if x < 2 else x)
data_objet_trouve_modifie["date"]= data_objet_trouve_modifie["date"].apply(lambda x : x[0:10] )
data_objet_trouve_modifie


# In[593]:


df_nombre_objet_trouve =data_objet_trouve_modifie
df_nombre_objet_trouve['date']= pd.to_datetime(df_nombre_objet_trouve['date'])
df_nombre_objet_trouve


# In[ ]:





# In[594]:


df_nombre_objet_trouve_semaine=df_nombre_objet_trouve.groupby(by="date").apply(lambda s: pd.Series({ 
    "nombre_d'objet trouvé": s["date"].count(), 

}))


# In[595]:


df_nombre_objet_trouve.dtypes


# In[596]:


df_nombre_objet_trouve_semaine


# In[597]:


df_nombre_objet_trouve_semaine.reset_index(inplace=True)


# In[598]:


df_nombre_objet_trouve_semaine


# In[599]:


df_semaine=df_nombre_objet_trouve_semaine.groupby(pd.Grouper(key='date', freq='W-MON'))["nombre_d'objet trouvé"].agg('sum')
df_semaine


# In[600]:


df_semaine = df_semaine.to_frame()
df_semaine


# In[601]:


df_semaine.hist()


# In[602]:


df_semaine.reset_index(inplace=True)


# In[603]:


df_semaine["date"][313]


# In[604]:


x=df_semaine["date"]
y=df_semaine["nombre_d\'objet trouvé"]
# Créez l'histogramme
plt.bar(x,y )

# Spécifiez les dates à afficher en Datetime
plt.xlim(df_semaine["date"][0], df_semaine["date"][313])

# # Ajoutez des étiquettes pour chaque barre
# plt.set_xticklabels(x)

# Affichez le graphique
plt.show()


# Pour mieux presenter
# 
# plt.figure(figsize=(12,6))
# 
# quartier= list(df_air_bnb_anvers.neighbourhood_cleansed.unique())
# x_pos = np.arange(len(quartier))
# 
# plt.bar(x_pos, counts)
# plt.xticks(x_pos, quartier, rotation=90)
# 
# 
# plt.show()

# Question n°2

# In[605]:


#la même question , mais il faut afficher seur le même graphique en fonction du type d'objet


# In[606]:


df_nombre_objet_trouve


# In[607]:


df_nombre_objet_trouve_evolution=df_nombre_objet_trouve.groupby(by=["date","type_objet_trouve"]).apply(lambda s: pd.Series({ 
    "nombre_d'objet trouvé": s["date"].count(), 

}))


# In[608]:


df_nombre_objet_trouve_evolution.reset_index(inplace=True)
df_nombre_objet_trouve_evolution


# In[609]:


df_list_objet = df_nombre_objet_trouve_evolution["type_objet_trouve"].unique()


# In[610]:


list_objet_trouve = list(df_list_objet)


# In[611]:


list_objet_trouve[0]


# In[612]:


df_nombre_objet_trouve_evolution["nombre_d\'objet trouvé"]


# In[ ]:





# In[613]:


df_nombre_objet_trouve_evolution_1 = df_nombre_objet_trouve_evolution[df_nombre_objet_trouve_evolution["type_objet_trouve"]==list_objet_trouve[0]]
df_nombre_objet_trouve_evolution_1 = df_nombre_objet_trouve_evolution_1.drop(["type_objet_trouve"], axis=1)


# In[614]:


df_nombre_objet_trouve_evolution_2 = df_nombre_objet_trouve_evolution[df_nombre_objet_trouve_evolution["type_objet_trouve"]==list_objet_trouve[1]]
df_nombre_objet_trouve_evolution_2 = df_nombre_objet_trouve_evolution_2.drop(["type_objet_trouve"], axis=1)


# In[615]:


df_nombre_objet_trouve_evolution_3 = df_nombre_objet_trouve_evolution[df_nombre_objet_trouve_evolution["type_objet_trouve"]==list_objet_trouve[2]]
df_nombre_objet_trouve_evolution_3 = df_nombre_objet_trouve_evolution_3.drop(["type_objet_trouve"], axis=1)


# In[616]:


df_nombre_objet_trouve_evolution_4 = df_nombre_objet_trouve_evolution[df_nombre_objet_trouve_evolution["type_objet_trouve"]==list_objet_trouve[3]]
df_nombre_objet_trouve_evolution_4 = df_nombre_objet_trouve_evolution_4.drop(["type_objet_trouve"], axis=1)


# In[617]:


df_nombre_objet_trouve_evolution_5 = df_nombre_objet_trouve_evolution[df_nombre_objet_trouve_evolution["type_objet_trouve"]==list_objet_trouve[4]]
df_nombre_objet_trouve_evolution_5 = df_nombre_objet_trouve_evolution_5.drop(["type_objet_trouve"], axis=1)


# In[618]:


df_nombre_objet_trouve_evolution_1


# In[619]:


df_nombre_objet_trouve_evolution_1.plot.hist(bins=20, alpha=0.5)


# In[620]:


bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:grey', 'tab:purple']


# In[621]:


ax1 = df_nombre_objet_trouve_evolution_1.plot.hist(bins=20, alpha=0.5,color=bar_colors[0])
df_nombre_objet_trouve_evolution_2.plot.hist(bins=20, alpha=0.5, ax=ax1,color=bar_colors[1])
df_nombre_objet_trouve_evolution_3.plot.hist(bins=20, alpha=0.5, ax=ax1,color=bar_colors[2])
df_nombre_objet_trouve_evolution_4.plot.hist(bins=20, alpha=0.5, ax=ax1,color=bar_colors[3])
df_nombre_objet_trouve_evolution_5.plot.hist(bins=20, alpha=0.5, ax=ax1,color=bar_colors[4])



# In[622]:


# plt.plot(x1,y1,alpha=0.5,color=bar_colors[0],label='Histogramme 1')
# plt.plot(x2,y2,alpha=0.5,color=bar_colors[1],label='Histogramme 2')
# plt.plot(x3,y3,alpha=0.5,color=bar_colors[2],label='Histogramme 3')
# plt.plot(x4,y4,alpha=0.5,color=bar_colors[3],label='Histogramme 4')
# plt.plot(x5,y5,alpha=0.5,color=bar_colors[4],label='Histogramme 5')

# # Affichez le graphique
# plt.show()



# In[623]:


plt.figure(figsize=(15,15))
x1=df_nombre_objet_trouve_evolution_1["date"]
y1=df_nombre_objet_trouve_evolution_1["nombre_d\'objet trouvé"]

x2=df_nombre_objet_trouve_evolution_2["date"]
y2=df_nombre_objet_trouve_evolution_2["nombre_d\'objet trouvé"]

x3=df_nombre_objet_trouve_evolution_3["date"]
y3=df_nombre_objet_trouve_evolution_3["nombre_d\'objet trouvé"]

x4=df_nombre_objet_trouve_evolution_4["date"]
y4=df_nombre_objet_trouve_evolution_4["nombre_d\'objet trouvé"]

x5=df_nombre_objet_trouve_evolution_5["date"]
y5=df_nombre_objet_trouve_evolution_5["nombre_d\'objet trouvé"]


# set the color used for the 5 bars
bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:grey', 'tab:purple']



plt.plot(x1,y1,alpha=0.5,color=bar_colors[0],label='Histogramme 1')
plt.plot(x2,y2,alpha=0.5,color=bar_colors[1],label='Histogramme 2')
plt.plot(x3,y3,alpha=0.5,color=bar_colors[2],label='Histogramme 3')
plt.plot(x4,y4,alpha=0.5,color=bar_colors[3],label='Histogramme 4')
plt.plot(x5,y5,alpha=0.5,color=bar_colors[4],label='Histogramme 5')

# Affichez le graphique
plt.show()


# In[624]:


plt.figure(figsize=(15,15))
x1=df_nombre_objet_trouve_evolution_1["date"]
y1=df_nombre_objet_trouve_evolution_1["nombre_d\'objet trouvé"]

x2=df_nombre_objet_trouve_evolution_2["date"]
y2=df_nombre_objet_trouve_evolution_2["nombre_d\'objet trouvé"]

x3=df_nombre_objet_trouve_evolution_3["date"]
y3=df_nombre_objet_trouve_evolution_3["nombre_d\'objet trouvé"]

x4=df_nombre_objet_trouve_evolution_4["date"]
y4=df_nombre_objet_trouve_evolution_4["nombre_d\'objet trouvé"]

x5=df_nombre_objet_trouve_evolution_5["date"]
y5=df_nombre_objet_trouve_evolution_5["nombre_d\'objet trouvé"]

# set the color used for the 5 bars
bar_colors_plotly = ['red', 'blue', 'green', 'grey', 'purple']

fig1 = px.line(df_nombre_objet_trouve_evolution_1, x1, y1, title=list_objet_trouve[0])
fig2 = px.line(df_nombre_objet_trouve_evolution_2, x2 , y2, title=list_objet_trouve[1])
fig3 = px.line(df_nombre_objet_trouve_evolution_3, x3, y3, title=list_objet_trouve[2])
fig4 = px.line(df_nombre_objet_trouve_evolution_4, x4, y4, title=list_objet_trouve[3])
fig5 = px.line(df_nombre_objet_trouve_evolution_5, x5, y5, title=list_objet_trouve[4])



# Créez l'histogramme
fig1.update_traces(line=dict(color=bar_colors_plotly[0]))
fig1.show()

fig2.update_traces(line=dict(color=bar_colors_plotly[1]))
fig2.show()

fig3.update_traces(line=dict(color=bar_colors_plotly[2]))
fig3.show()

fig4.update_traces(line=dict(color=bar_colors_plotly[3]))
fig4.show()

fig5.update_traces(line=dict(color=bar_colors_plotly[4]))
fig5.show()
# # Spécifiez les dates à afficher en Datetime
# plt.xlim(df_nombre_objet_trouve_evolution_1["date"][0], df_nombre_objet_trouve_evolution_1["date"][2147])


# In[625]:


df_semaine_evolution_1=df_nombre_objet_trouve_evolution_1.groupby(pd.Grouper(key='date', freq='W-MON'))["nombre_d'objet trouvé"].agg('sum')
df_semaine_evolution_1=df_semaine_evolution_1.to_frame()


# In[626]:


df_semaine_evolution_1.reset_index(inplace=True)


# In[627]:


df_semaine_evolution_1


# In[628]:


df_semaine_evolution_2=df_nombre_objet_trouve_evolution_2.groupby(pd.Grouper(key='date', freq='W-MON'))["nombre_d'objet trouvé"].agg('sum')
df_semaine_evolution_2= df_semaine_evolution_2.to_frame()


# In[629]:


df_semaine_evolution_2.reset_index(inplace=True)


# In[630]:


df_semaine_evolution_3=df_nombre_objet_trouve_evolution_3.groupby(pd.Grouper(key='date', freq='W-MON'))["nombre_d'objet trouvé"].agg('sum')
df_semaine_evolution_3 = df_semaine_evolution_3.to_frame()


# In[631]:


df_semaine_evolution_3.reset_index(inplace=True)


# In[632]:


df_semaine_evolution_4=df_nombre_objet_trouve_evolution_4.groupby(pd.Grouper(key='date', freq='W-MON'))["nombre_d'objet trouvé"].agg('sum')
df_semaine_evolution_4 = df_semaine_evolution_4.to_frame()


# In[633]:


df_semaine_evolution_4.reset_index(inplace=True)


# In[634]:


df_semaine_evolution_5=df_nombre_objet_trouve_evolution_5.groupby(pd.Grouper(key='date', freq='W-MON'))["nombre_d'objet trouvé"].agg('sum')
df_semaine_evolution_5 = df_semaine_evolution_5.to_frame()


# In[635]:


df_semaine_evolution_5.reset_index(inplace=True)


# In[636]:


plt.figure(figsize=(15,15))
x1_1=df_semaine_evolution_1["date"]
y1_1=df_semaine_evolution_1["nombre_d\'objet trouvé"]

x2_1=df_semaine_evolution_2["date"]
y2_1=df_semaine_evolution_2["nombre_d\'objet trouvé"]

x3_1=df_semaine_evolution_3["date"]
y3_1=df_semaine_evolution_3["nombre_d\'objet trouvé"]

x4_1=df_semaine_evolution_4["date"]
y4_1=df_semaine_evolution_4["nombre_d\'objet trouvé"]

x5_1=df_semaine_evolution_5["date"]
y5_1=df_semaine_evolution_5["nombre_d\'objet trouvé"]

# set the color used for the 5 bars
bar_colors_plotly = ['red', 'blue', 'green', 'grey', 'purple']

fig1 = px.line(df_semaine_evolution_1, x1_1, y1_1, title=list_objet_trouve[0])
fig2 = px.line(df_semaine_evolution_2, x2_1 , y2_1, title=list_objet_trouve[1])
fig3 = px.line(df_semaine_evolution_3, x3_1, y3_1, title=list_objet_trouve[2])
fig4 = px.line(df_semaine_evolution_4, x4_1, y4_1, title=list_objet_trouve[3])
fig5 = px.line(df_semaine_evolution_5, x5_1, y5_1, title=list_objet_trouve[4])



# Créez l'histogramme
fig1.update_traces(line=dict(color=bar_colors_plotly[0]))
fig1.show()

fig2.update_traces(line=dict(color=bar_colors_plotly[1]))
fig2.show()

fig3.update_traces(line=dict(color=bar_colors_plotly[2]))
fig3.show()

fig4.update_traces(line=dict(color=bar_colors_plotly[3]))
fig4.show()

fig5.update_traces(line=dict(color=bar_colors_plotly[4]))
fig5.show()


# In[637]:


fig, ax = plt.subplots()

x1_1=df_semaine_evolution_1["date"]
y1_1=df_semaine_evolution_1["nombre_d\'objet trouvé"]

x2_1=df_semaine_evolution_2["date"]
y2_1=df_semaine_evolution_2["nombre_d\'objet trouvé"]

x3_1=df_semaine_evolution_3["date"]
y3_1=df_semaine_evolution_3["nombre_d\'objet trouvé"]

x4_1=df_semaine_evolution_4["date"]
y4_1=df_semaine_evolution_4["nombre_d\'objet trouvé"]

x5_1=df_semaine_evolution_5["date"]
y5_1=df_semaine_evolution_5["nombre_d\'objet trouvé"]

# set the color used for the 5 bars
bar_colors_plotly = ['red', 'blue', 'green', 'grey', 'purple']

ax.plot(x1_1,y1_1, color=bar_colors_plotly[0], label=list_objet_trouve[0])
ax.plot(x2_1,y2_1, color=bar_colors_plotly[1], label=list_objet_trouve[1])
ax.plot(x3_1,y3_1, color=bar_colors_plotly[2], label=list_objet_trouve[2])
ax.plot(x4_1,y4_1, color=bar_colors_plotly[3], label=list_objet_trouve[3])
ax.plot(x5_1,y5_1, color=bar_colors_plotly[4], label=list_objet_trouve[4])
ax.legend()
ax.set_title('Evolution des objets trouvés par semaine')
ax.set_xlabel('Date')
ax.set_ylabel


# Question n°3

# In[ ]:





# In[638]:


import folium

# Création d'une carte centrée sur la France
m = folium.Map(location=[47.07, 2.39], zoom_start=6)

# Ajout des marqueurs sur la carte
folium.Marker([48.86, 2.35], popup='Paris').add_to(m)
folium.Marker([43.3, 5.4], popup='Marseille').add_to(m)
folium.Marker([44.83, -0.58], popup='Bordeaux').add_to(m)
folium.Marker([43.6, 1.4], popup='Toulouse').add_to(m)

# Affichage de la carte
m.fit_bounds([[41.3625, -5.9643], [51.09, 9.56]])
folium.Marker(location=[48.8566, 2.3522], popup='Paris').add_to(m)

m


# In[639]:


df_frequentation =pd.read_csv("frequentation_mensuel.csv")


# In[640]:


df_frequentation


# Partie data analyse en vue de la DATA SCIENCE. - (sur un notebook)++

# Question n°1

# In[ ]:





# In[641]:


df_analysis = pd.read_csv('merge_final.csv')


# In[642]:


df_analysis


# In[643]:


df_presentation = df_analysis

# Créer le scatterplot
fig = px.scatter(df_presentation, x='temperature_moyenne', y='nb_lost_item')

# Afficher le scatterplot
fig.show()


# In[644]:


fig = px.scatter_matrix(df_presentation, dimensions = ["temperature_moyenne","nb_lost_item"])
fig


# In[645]:


df_presentation=df_presentation.drop("Unnamed: 0", axis=1)
df_presentation


# In[646]:


df_presentation.corr()


# In[647]:


sns.heatmap(df_presentation.corr(), cmap='coolwarm')


# In[648]:


df_nombre_objet_trouve


# In[649]:


df_nombre_objet_trouve_saison_preparation=df_nombre_objet_trouve.groupby(by="date").apply(lambda s: pd.Series({ 
    "nombre_d'objet trouvé": s["date"].count(), 

}))


# In[650]:


df_nombre_objet_trouve_saison_preparation


# In[651]:


# df_nombre_objet_trouve_saison_preparation =df_nombre_objet_trouve_saison_preparation.reset_index(inplace=True)


# In[652]:


df_saison = df_nombre_objet_trouve_saison_preparation.resample("90D").median()


# In[653]:


df_saison


# In[654]:


df_saison.reset_index(inplace=True)


# In[655]:


df_saison


# question n°3
# 

# In[656]:


df_nombre_objet_trouve_saison_preparation


# In[657]:


df_saison_boxplot = df_nombre_objet_trouve_saison_preparation.resample("90D").mean()


# In[658]:


df_saison_boxplot


# In[659]:


df_saison_boxplot.shape


# In[660]:


liste_saison = ["Printemps","Eté","Automne","Hiver"]
liste_saison_objet_trouve =[ liste_saison[i%4] for i in range(0,25)]
liste_saison_objet_trouve


# In[661]:


df_saison_boxplot.loc[:, 'Saison'] = liste_saison_objet_trouve


# In[662]:


df_saison_boxplot


# In[663]:


df_saison_boxplot.reset_index(inplace=True)


# In[664]:


df_saison_boxplot


# In[665]:


df_saison_boxplot_final= df_saison_boxplot.drop("date",axis=1)


# In[666]:


df_saison_boxplot_final


# In[667]:


df_saison_boxplot_final.reset_index(inplace=True)


# In[668]:


fig = px.box(df_saison_boxplot_final, x="Saison", y="nombre_d'objet trouvé", title="Boxplot de mon dataframe",
              )

fig.show()


# In[669]:


df_nombre_objet_trouve_saison_preparation


# In[670]:


df_saison_boxplot_total = df_nombre_objet_trouve_saison_preparation


# In[671]:


df_nombre_objet_trouve_saison_preparation.shape


# In[672]:


df_saison_prerparation = df_nombre_objet_trouve_saison_preparation


# In[673]:


df_saison_prerparation.reset_index(inplace=True)


# In[674]:


liste_saison_total = list(df_saison_prerparation['date'].to_numpy())


# In[675]:


liste_saison_total


# In[676]:


from dateutil import parser
datetime_obj = parser.parse('2018-02-06T13:12:18.1278015Z')


# In[677]:


datetime_liste_saison_total = [parser.parse(str(dt)) for dt in liste_saison_total]


# In[678]:


datetime_liste_saison_total


# In[679]:


def get_season(day): 
  iso_year, iso_week, iso_day = day.isocalendar()

  if iso_week >= 1 and iso_week <= 13: 
    return 'Printemps'
  elif iso_week >= 14 and iso_week <= 26: 
    return 'Eté'
  elif iso_week >= 27 and iso_week <= 39: 
    return 'Automne'
  elif iso_week >= 40 and iso_week <= 53: 
    return 'Hiver'


# In[680]:


liste_saison_total_modifie = [get_season(x) for x in datetime_liste_saison_total]


# In[681]:


liste_saison_total_modifie


# In[682]:


df_saison_boxplot_total.loc[:, 'Saison'] = liste_saison_total_modifie


# In[683]:


df_saison_boxplot_total


# In[684]:


df_saison_boxplot_total.drop("date",axis=1)


# In[685]:


fig6 = px.box(df_saison_boxplot_total, x="Saison", y="nombre_d'objet trouvé", title="Boxplot de mon dataframe",
              )

fig6.show()


# In[686]:


f_oneway_result =  stats.f_oneway(df_saison_boxplot_total["nombre_d'objet trouvé"], df_saison_boxplot_total['Saison']=="Hiver",df_saison_boxplot_total['Saison']=="Printemps",df_saison_boxplot_total['Saison']=="Eté",df_saison_boxplot_total['Saison']=="Automne")


# In[687]:


f_oneway_result


# Question n°4

# In[688]:


df_mois_contigency= df_nombre_objet_trouve.groupby(by="type_objet_trouve").apply(lambda s: pd.Series({ 
    "nombre_d'objet trouvé": s["date"].count(), 
    "date" : pd.Grouper(key='date', freq='M')
}))


# In[689]:


df_mois_contigency


# In[690]:


df_mois_seulement= df_nombre_objet_trouve.groupby(by=["type_objet_trouve",pd.Grouper(key='date', freq='M')]).apply(lambda s: pd.Series({ 
    "nombre_d'objet trouvé": s["date"].count(), 

}))


# In[691]:


# df_mois_seulement.reset_index(inplace=True)


# In[692]:


df_mois_seulement


# In[693]:


df_mois_contigency


# In[694]:


# df_mois_preparation.reset_index(inplace=True)


# In[696]:


chi2, p, dof, expected = chi2_contingency(df_mois_seulement)


# In[697]:


chi2

