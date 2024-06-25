import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from .forms import ForquilhinhaPassadoForm, ForquilhinhaFuturoForm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

matplotlib.use('Agg') #Usado para salvar nosso gráfico em PNG (Agg é usado geralmente em GUI, onde nao há visualizaçao gráfica)

csv_df1 = os.path.join(settings.BASE_DIR, 'suslog_project/static/datasets/suslogbr_df1.csv')

df = pd.read_csv(csv_df1)
df['Data'] = pd.to_datetime(df['Data'])
df['Vacinas aplicadas'] = df['Vacinas aplicadas'].str.replace(',', '').astype(float)
df['Mes'] = df['Data'].dt.month
df['Ano'] = df['Data'].dt.year
df['Trimestre'] = df['Data'].dt.quarter

X = df[['Mes', 'Ano', 'Trimestre']]
y = df['Vacinas aplicadas']

# Normalizando nossas features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar: Random Forest (melhor modelo testado para regressao)
rf = RandomForestRegressor(random_state=42)

# Usando GridSearchCV para testar os melhores parâmetros automaticamente
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 30, 50],
    'max_features': ['sqrt', 10],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Melhores parametros encontrados: ", best_params)

# Treinando RF com os melhores parametros
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

def compare_year(request):
    form = ForquilhinhaPassadoForm(request.POST or None)
    data_year = pd.DataFrame()
    graph_path = None
    has_data = False
    if form.is_valid():
        year_to_compare = form.cleaned_data['Ano']
        data_year = df[df['Ano'] == year_to_compare].copy()
        if not data_year.empty:
            data_year_scaled = scaler.transform(data_year[['Mes', 'Ano', 'Trimestre']])
            predictions_year = best_rf.predict(data_year_scaled)
            data_year['Vacinas previstas'] = predictions_year
            data_year.reset_index(drop=True, inplace=True)
            data_year.index += 1
            data_year.rename(columns={'Data': 'data', 'Vacinas aplicadas': 'vacinas_aplicadas', 'Vacinas previstas': 'vacinas_previstas'}, inplace=True)
            has_data = True

            # Gerando o gráfico
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data_year, x='data', y='vacinas_aplicadas', label='Real')
            sns.lineplot(data=data_year, x='data', y='vacinas_previstas', label='Predito')
            plt.xticks(rotation=45)
            plt.title(f'Real vs Vacinas preditas em {year_to_compare}')
            plt.xlabel('Data')
            plt.ylabel('Número de vacinas')
            plt.legend()

            # Salvar o gráfico
            graph_filename = f'graph_{year_to_compare}.png'
            media_dir = os.path.join(settings.BASE_DIR, 'suslog_project/static/images')
            graph_path = os.path.join(media_dir, graph_filename)
            plt.savefig(graph_path)
            plt.close()
            graph_path = f'images/{graph_filename}'
                 
    return render(request, 'predicao/passado.html', {'form': form, 'data_year': data_year, 'has_data': has_data, 'graph_path': graph_path})

def generate_future_dates(start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    future_df = pd.DataFrame({'Data': future_dates})
    future_df['Mes'] = future_df['Data'].dt.month
    future_df['Ano'] = future_df['Data'].dt.year  
    future_df['Trimestre'] = future_df['Data'].dt.quarter
    return future_df

def future_predictions(request):
    form = ForquilhinhaFuturoForm(request.POST or None)
    future_df = pd.DataFrame()
    graph_path = None
    has_data = False
    if form.is_valid():
        year = form.cleaned_data['year']
        future_df = generate_future_dates(f'{year}-01-01', f'{year}-12-01')
        future_features = scaler.transform(future_df[['Mes', 'Ano', 'Trimestre']])
        future_predictions_rf = best_rf.predict(future_features)
        future_df['Vacinas_RF'] = future_predictions_rf
        has_data = True

        # Gerando gráfico
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=future_df, x='Data', y='Vacinas_RF', label='Random Forest')
        plt.xticks(rotation=45)
        plt.title(f'Prediçoes futuras de vacinas aplicadas em {year}')
        plt.xlabel('Data')
        plt.ylabel('Número de vacinas')
        plt.legend()

        # Salvar gráfico
        graph_filename = f'predicao_futura_{year}.png'
        media_dir = os.path.join(settings.BASE_DIR, 'suslog_project/static/images')
        graph_path = os.path.join(media_dir, graph_filename)
        plt.savefig(graph_path)
        plt.close()
        graph_path = f'images/{graph_filename}'
    
    return render(request, 'predicao/futuro.html', {'form': form, 'future_df': future_df, 'has_data': has_data, 'graph_path': graph_path})
