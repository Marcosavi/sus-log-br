import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from .forms import CompararModelsForm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use('Agg')

df = pd.read_csv("suslog_project/static/datasets/suslogbr_df1.csv")
df['Data'] = pd.to_datetime(df['Data'])
df['Vacinas aplicadas'] = df['Vacinas aplicadas'].str.replace(',', '').astype(float)
df['Mes'] = df['Data'].dt.month
df['Ano'] = df['Data'].dt.year
df['Trimestre'] = df['Data'].dt.quarter

X = df[['Mes', 'Ano', 'Trimestre']]
y = df['Vacinas aplicadas']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=4, min_samples_leaf=4, random_state=42)
rf.fit(X_train, y_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

svr = SVR()

grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, n_jobs=1, verbose=2, scoring='neg_mean_squared_error')

grid_search_svr.fit(X_train, y_train)

best_params_svr = grid_search_svr.best_params_
print("Melhores parâmetros para SVR: ", best_params_svr)

best_svr = grid_search_svr.best_estimator_
best_svr.fit(X_train, y_train)

def compare_models_df1(request):
    form = CompararModelsForm(request.POST or None)
    data_year = pd.DataFrame()
    graph_path = None
    has_data = False
    metrics = {}
    if form.is_valid():
        year_to_compare = form.cleaned_data['Ano']
        data_year = df[df['Ano'] == int(year_to_compare)].copy()
        if not data_year.empty:
            data_year_scaled = scaler.transform(data_year[['Mes', 'Ano', 'Trimestre']])

            predictions_rf = rf.predict(data_year_scaled)
            predictions_lr = lr.predict(data_year_scaled)
            predictions_svm = best_svr.predict(data_year_scaled)

            data_year['Vacinas_RF'] = predictions_rf
            data_year['Vacinas_LR'] = predictions_lr
            data_year['Vacinas_SVM'] = predictions_svm

            data_year.reset_index(drop=True, inplace=True)
            data_year.index += 1
            data_year.rename(columns={'Data': 'data', 'Vacinas aplicadas': 'vacinas_aplicadas'}, inplace=True)
            has_data = True

            # Calcular Métricas
            metrics['RF_MAE'] = mean_absolute_error(data_year['vacinas_aplicadas'], data_year['Vacinas_RF'])
            metrics['RF_MSE'] = mean_squared_error(data_year['vacinas_aplicadas'], data_year['Vacinas_RF'])
            metrics['LR_MAE'] = mean_absolute_error(data_year['vacinas_aplicadas'], data_year['Vacinas_LR'])
            metrics['LR_MSE'] = mean_squared_error(data_year['vacinas_aplicadas'], data_year['Vacinas_LR'])
            metrics['SVM_MAE'] = mean_absolute_error(data_year['vacinas_aplicadas'], data_year['Vacinas_SVM'])
            metrics['SVM_MSE'] = mean_squared_error(data_year['vacinas_aplicadas'], data_year['Vacinas_SVM'])

            # Gerando o gráfico
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data_year, x='data', y='vacinas_aplicadas', label='Real')
            sns.lineplot(data=data_year, x='data', y='Vacinas_RF', label='Predito | Floresta aleatória')
            sns.lineplot(data=data_year, x='data', y='Vacinas_LR', label='Predito | Regressao linear')
            sns.lineplot(data=data_year, x='data', y='Vacinas_SVM', label='Predito | SVM')
            plt.xticks(rotation=45)
            plt.title(f'Comparaçao para {year_to_compare}')
            plt.xlabel('Data')
            plt.ylabel('Número de vacinas aplicadas')

            # Salvar o gráfico
            graph_filename = f'comparar_modelos_{year_to_compare}.png'
            media_dir = os.path.join(settings.BASE_DIR, 'suslog_project/static/images')
            graph_path = os.path.join(media_dir, graph_filename)
            plt.savefig(graph_path)
            plt.close()
            graph_path = f'images/{graph_filename}'
            
    return render(request, 'comparar_modelos/index.html', {'form': form, 'data_year': data_year, 'has_data': has_data, 'graph_path': graph_path, 'metrics': metrics})
