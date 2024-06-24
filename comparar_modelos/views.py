import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from .forms import CompareModelsForm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Use the Agg backend for non-interactive plotting
matplotlib.use('Agg')

# Define the path to the CSV file
csv_path = os.path.join(settings.BASE_DIR, 'suslog_project/static/csv_files/suslogbr_df1.csv')

# Load and preprocess the dataset
df = pd.read_csv(csv_path)
df['Data'] = pd.to_datetime(df['Data'])
df['Vacinas aplicadas'] = df['Vacinas aplicadas'].str.replace(',', '').astype(float)
df['Mes'] = df['Data'].dt.month
df['Ano'] = df['Data'].dt.year
df['Dia'] = df['Data'].dt.day
df['Dia_da_semana'] = df['Data'].dt.dayofweek
df['Trimestre'] = df['Data'].dt.quarter

X = df[['Mes', 'Ano', 'Dia', 'Dia_da_semana', 'Trimestre']]
y = df['Vacinas aplicadas']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=4, min_samples_leaf=4, random_state=42)
rf.fit(X_train, y_train)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train SVM
svm = SVR(kernel='rbf', C=1, epsilon=0.1)
svm.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Define the parameter grid for SVR
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

# Initialize the SVR
svr = SVR()

# Initialize GridSearchCV for SVR
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search_svr.fit(X_train, y_train)

# Get the best parameters
best_params_svr = grid_search_svr.best_params_
print("Melhores parâmetros para SVR: ", best_params_svr)

# Train the SVR with the best parameters
best_svr = grid_search_svr.best_estimator_
best_svr.fit(X_train, y_train)


def compare_models_df1(request):
    form = CompareModelsForm(request.POST or None)
    data_year = pd.DataFrame()
    graph_path = None
    has_data = False
    metrics = {}
    if form.is_valid():
        year_to_compare = form.cleaned_data['Ano']
        data_year = df[df['Ano'] == int(year_to_compare)].copy()
        if not data_year.empty:
            data_year_scaled = scaler.transform(data_year[['Mes', 'Ano', 'Dia', 'Dia_da_semana', 'Trimestre']])
            predictions_rf = rf.predict(data_year_scaled)
            predictions_lr = lr.predict(data_year_scaled)
            predictions_svm = svm.predict(data_year_scaled)
            data_year['Vacinas_RF'] = predictions_rf
            data_year['Vacinas_LR'] = predictions_lr
            data_year['Vacinas_SVM'] = predictions_svm
            data_year.reset_index(drop=True, inplace=True)  # Reset index to avoid index issues
            data_year.index += 1  # Start index at 1
            data_year.rename(columns={'Data': 'data', 'Vacinas aplicadas': 'vacinas_aplicadas'}, inplace=True)
            has_data = True

            # Calculate metrics
            metrics['RF_MAE'] = mean_absolute_error(data_year['vacinas_aplicadas'], data_year['Vacinas_RF'])
            metrics['RF_MSE'] = mean_squared_error(data_year['vacinas_aplicadas'], data_year['Vacinas_RF'])
            metrics['LR_MAE'] = mean_absolute_error(data_year['vacinas_aplicadas'], data_year['Vacinas_LR'])
            metrics['LR_MSE'] = mean_squared_error(data_year['vacinas_aplicadas'], data_year['Vacinas_LR'])
            metrics['SVM_MAE'] = mean_absolute_error(data_year['vacinas_aplicadas'], data_year['Vacinas_SVM'])
            metrics['SVM_MSE'] = mean_squared_error(data_year['vacinas_aplicadas'], data_year['Vacinas_SVM'])

            # Ensure the directory for saving the graph exists
            media_dir = os.path.join(settings.BASE_DIR, 'suslog_project/static/images')
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Generate the graph
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data_year, x='data', y='vacinas_aplicadas', label='Real')
            sns.lineplot(data=data_year, x='data', y='Vacinas_RF', label='Predito | Floresta aleatória')
            sns.lineplot(data=data_year, x='data', y='Vacinas_LR', label='Predito | Regressao linear')
            sns.lineplot(data=data_year, x='data', y='Vacinas_SVM', label='Predito | SVM')
            plt.xticks(rotation=45)
            plt.title(f'Comparaçao para {year_to_compare}')
            plt.xlabel('Data')
            plt.ylabel('Número de vacinas aplicadas')
            plt.legend()

            # Save the plot
            graph_filename = f'comparar_modelos_{year_to_compare}.png'
            graph_path = os.path.join(media_dir, graph_filename)
            plt.savefig(graph_path)
            plt.close()

            # Update graph_path to be relative to the static directory
            graph_path = f'images/{graph_filename}'
            
    return render(request, 'comparar_modelos/index.html', {'form': form, 'data_year': data_year, 'has_data': has_data, 'graph_path': graph_path, 'metrics': metrics})
