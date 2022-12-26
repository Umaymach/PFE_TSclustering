
#All the packages that are used 
import pandas as pd
import requests
import csv
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
import pandas as pd
import numpy as np
import sklearn
import tslearn.clustering
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from datetime import datetime as dt

#Collecting the data using API by specifing longitude and latitude and saving the file as csv
def datacollection(longitude, latitude):
    output = r""
    url = r"https://power.larc.nasa.gov/api/temporal/daily/point?start=1981&end=2021&longitude={longitude}&latitude={latitude}&community=ag&parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,QV2M,WS10M&format=csv&header=False"
    request = url.format(longitude=longitude, latitude=latitude)
    response = requests.get(url=request, verify=True, timeout=30.00)
    open(f'Data_{latitude,longitude}.csv', 'wb').write(response.content)
    data= pd.read_csv(f'Data_{latitude,longitude}.csv')
    return data

#Preparing the format of the data 
def formatpreparing(data):
    data=data.rename(columns={'YEAR': 'Année'})
    data["combined"] = data["Année"]*1000 + data["DOY"]
    data["Date"] = pd.to_datetime(data["combined"], format = "%Y%j")
    data['Jour-Mois']=data['Date'].dt.strftime('%d-%m')
    data=data.set_index("Date")
    data=data[['Année',"T2M_MAX",'T2M_MIN','PRECTOTCORR','QV2M','WS10M','Jour-Mois']]
    return data

#Organizing the data by the specific culture's season
def prepare_data(data, season_debute_month,season_debute_day, season_end_month,season_end_day):
    #take only the agricultural season that we working with for the specific culture
    data=data[(data.index.month >= season_debute_month) | (data.index.month <= season_end_month)]
    data=data[data.index >=dt(1981,season_debute_month,season_debute_day)]
    #create the column "saison agricole"
    years = data["Année"].unique()
    data["Saison agricole"] = [0 for i in range(data.shape[0])]
    for year in years:
        data.loc[(data.index >= dt(year, season_debute_month, season_debute_day)) & (data.index <= dt(year+1, season_end_month, season_end_day)), "Saison agricole"] = f"{year}-{year+1}"
    data=data[["T2M_MAX",'T2M_MIN','PRECTOTCORR','QV2M','WS10M','Saison agricole']]
    data=data.reset_index()
    #create column "jour mois"
    data['Jour-Mois']=data["Date"].dt.strftime('%d-%m')
    return data

#Calculating the AGDD and APRE
def calculat_AGDD_APRE(data, culture, Tbase):
    data[f'GDD de {culture}']=(data["T2M_MAX"]+data["T2M_MIN"])/2-Tbase
    data[f'GDD de {culture}'] = data[f'GDD de {culture}'].clip(lower = 0)
    data['AGDD'] = data.groupby(['Saison agricole'])[f'GDD de {culture}'].transform(pd.Series.cumsum)
    data['APRE'] = data.groupby(['Saison agricole'])['PRECTOTCORR'].transform(pd.Series.cumsum)
    data=data[["Date","AGDD",'APRE','QV2M','WS10M','Jour-Mois','Saison agricole']]
    data=data.set_index('Saison agricole')
    return data

#Studying the clustering of AGDD
def pivot_with_AGDD(data):
    cols=data["Jour-Mois"].unique().tolist()
    data= data.pivot_table(index="Saison agricole", columns="Jour-Mois", values="AGDD")[cols[:-1]]
    data=data.drop('2021-2022')
    return data

#Studying the clustering of APRE
def pivot_with_APRE(data):
    cols=data["Jour-Mois"].unique().tolist()
    data= data.pivot_table(index="Saison agricole", columns="Jour-Mois", values="APRE")[cols[:-1]]
    data=data.drop('2021-2022')
    return data

#Studying the clustering of QV2M
def pivot_with_QV2M(data):
    cols=data["Jour-Mois"].unique().tolist()
    data= data.pivot_table(index="Saison agricole", columns="Jour-Mois", values="QV2M")[cols[:-1]]
    data=data.drop('2021-2022')
    return data

#Studying the clustering of APRE
def pivot_with_WS10M(data):
    cols=data["Jour-Mois"].unique().tolist()
    data= data.pivot_table(index="Saison agricole", columns="Jour-Mois", values="WS10M")[cols[:-1]]
    data=data.drop('2021-2022')
    return data


#Plotting the seasons of each culture
def Plotting_seasons(data):
    TAGDD=pivot_with_AGDD(data)
    TAPRE= pivot_with_APRE(data)
    TQV2M=pivot_with_QV2M(data)
    TWS10M=pivot_with_WS10M(data)
    
    for var, i in [(TAGDD,"AGDD"),(TAPRE, "APRE"),(TQV2M,"QV2M"), (TWS10M,"WS10M")]:
        ax = var.T.plot(figsize=(20, 12))
        ax.set_ylabel(i, fontsize=20)
        ax.set_xlabel('Jour et mois', fontsize=20)
    return


#Calculating the minmax scaling for the data base
def minmax(data):
    datanorm = (data-min(data.min()))/(max(data.max())-min(data.min()))
    return datanorm


#Searching the elbow and the number of the cluters, then predicting 
def Clustering_with_timeseriesKmeans(data_scaled):
    metric_params = {"global_constraint":"sakoe_chiba", "sakoe_chiba_radius": 10}
    vis=kelbow_visualizer(TimeSeriesKMeans(random_state=42), data_scaled, k=(2,10), metric_params=metric_params,locate_elbow=True, timings=False, show=True)
    vis.fit(data_scaled)
    num_K=vis.elbow_value_
    models = tslearn.clustering.TimeSeriesKMeans(n_clusters=num_K, metric='dtw',random_state=42, metric_params=metric_params)
    predictions = models.fit_predict(data_scaled)
    return num_K, predictions

#Display the plots and the clusters by groups
def Clusteringplots(data_scaled,T):
    metric_params = {"global_constraint":"sakoe_chiba", "sakoe_chiba_radius": 10}
    vis=kelbow_visualizer(TimeSeriesKMeans(random_state=42), data_scaled, k=(2,10), metric_params=metric_params,locate_elbow=True, timings=False, show=True)
    vis.fit(data_scaled)
    num_K=vis.elbow_value_
    # sakoe_chiba_radius=None, itakura_max_slope=None
    models = tslearn.clustering.TimeSeriesKMeans(n_clusters=num_K, metric='dtw',random_state=42, metric_params=metric_params)
    predictions = models.fit_predict(data_scaled)
    plt.figure(figsize=(20,10))

    #plt.figure(figsize=(20,10))
    X_train = data_scaled.values
    for yi in range(4):
        plt.subplot(2, 2, yi + 1)
        for xx in X_train[predictions == yi]:
            #T the pivoted data base=pivot_with_AGDD(d)
            _index = T.columns.values
            n_indices = _index.shape[0]
            _index = [_index[i] for i in range(n_indices) if i%31==0 ]
            plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.xticks(ticks = [i for i in range(n_indices) if  i%31==0], labels = _index)
        plt.plot(models.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, X_train.shape[1])
        # plt.ylim(-10, 10)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
    data_scaled["cluster"] = predictions

    print(f"the number optimal of classes is:{num_K}")
    print('Cluster 1 :', list(data_scaled[data_scaled.cluster == 0].index))
    print('Cluster 2 :', list(data_scaled[data_scaled.cluster == 1].index))
    print('Cluster 3 :', list(data_scaled[data_scaled.cluster == 2].index))
    print('Cluster 4 :', list(data_scaled[data_scaled.cluster == 3].index))
    return


#Calculating the silhoute score for the predictions of the last time series clustering 
def silhouette_score_fct(data_scaled):
    predictions=Clustering_with_timeseriesKmeans(data_scaled)
    metric_params = {"global_constraint":"sakoe_chiba", "sakoe_chiba_radius": 10}
    s=tslearn.clustering.silhouette_score(data_scaled,predictions[1], metric="dtw",random_state=5, metric_params=metric_params) 
    return s

results = {}
#Calculating the calinski harabasz score for the predictions of the last time series clustering 
def calinski_harabasz_score(data_scaled):
    predictions=Clustering_with_timeseriesKmeans(data_scaled)
    metric_params = {"global_constraint":"sakoe_chiba", "sakoe_chiba_radius": 10}
    s=sklearn.metrics.calinski_harabasz_score(data_scaled,predictions[1] )
    return s


#Calculating the means that going to be used in the second clustering for AGDD
def means_AGDD(d):
    pd.set_option('max_columns', 6)
    predictionsT=Clustering_with_timeseriesKmeans(minmax(pivot_with_AGDD(d)))

    X=pivot_with_AGDD(d)
    X["cluster"] = predictionsT[1]

    Tcarac = {i: X[X.cluster == i] for i in X.cluster}
    for i in X.cluster :
        Tcarac[i]['Max']=Tcarac[i].iloc[:, 0:366].max(axis=1)  
    agdd_means=[]
    for i in X.cluster:
        agdd_means.append(Tcarac[i][['Max']].mean(axis=0).item())
    
    agdd_means=pd.DataFrame(agdd_means,columns = ['La moyenne de l\'AGDD pour chaque cluster'],index=X.index)
    agdd_means=agdd_means.join(X[["cluster"]])
    
    return agdd_means

#Calculating the means that going to be used in the second clustering for APRE
def means_APRE(d):
    pd.set_option('max_columns', 6)
    predictionsP=Clustering_with_timeseriesKmeans(minmax(pivot_with_APRE(d)))

    Y=pivot_with_APRE(d)
    Y["cluster"] = predictionsP[1]

    Pcarac = {i: Y[Y.cluster == i] for i in Y.cluster}
    for i in Y.cluster :
        Pcarac[i]['Max']=Pcarac[i].iloc[:, 0:366].max(axis=1)  
    apre_means=[]
    for i in Y.cluster:
        apre_means.append(Pcarac[i][['Max']].mean(axis=0).item())
    
    apre_means=pd.DataFrame(apre_means,columns = ['La moyenne de l\'APRE pour chaque cluster'],index=Y.index)
    apre_means=apre_means.join(Y[["cluster"]])
    
    return apre_means

#Grouping the means for the final data
def final_data():
    agdd_means=means_AGDD()
    apre_means=means_APRE()
    Final_data=pd.DataFrame(agdd_means["La moyenne de l'AGDD pour chaque cluster"])
    Final_data["La moyenne de l'APRE pour chaque cluster"]=apre_means["La moyenne de l'APRE pour chaque cluster"]
    return Final_data


# Scaling the final data fpr the second clustering
def minmaxscaling_final():
    scaler = MinMaxScaler()
    Final_data=final_data()
    col = Final_data.columns
    dfscaled = scaler.fit_transform(Final_data)
    dfscaled = pd.DataFrame(dfscaled,columns=col, index=Final_data.index)
    return dfscaled

# The final clustering and the result of the groups
def finalclustering():
    dfscaled=minmaxscaling_final()
    Final_data=final_data()
    visualizer=kelbow_visualizer(KMeans(random_state=42), dfscaled, k=(2,10),locate_elbow=True, timings=False)
    NC= visualizer.elbow_value_
    models = KMeans(n_clusters=NC,random_state=42)
    predictions = models.fit_predict(Final_data)
    print(predictions)
    Final_data["Final_Clustering"]=models.labels_
    print('Cluster 1 :', list(Final_data[Final_data.Final_Clustering == 0].index))
    print('Cluster 2 :', list(Final_data[Final_data.Final_Clustering == 1].index))
    print('Cluster 3 :', list(Final_data[Final_data.Final_Clustering == 2].index))
    print('Cluster 4 :', list(Final_data[Final_data.Final_Clustering == 3].index))
    print('Cluster 5 :', list(Final_data[Final_data.Final_Clustering == 4].index))
    return