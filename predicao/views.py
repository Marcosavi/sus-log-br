import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from .forms import ForquilhinhaVacinaForm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 30, 50],
    'max_features': ['sqrt', 10],
    'min_samples_split': [None, 4],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=0)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

def compare_year(request):
    form = ForquilhinhaVacinaForm(request.POST or None)
    data_year = pd.DataFrame()
    graph_path = None
    has_data = False
    if form.is_valid():
        year_to_compare = form.cleaned_data['Ano']
        data_year = df[df['Ano'] == year_to_compare].copy()
        if not data_year.empty:
            data_year_scaled = scaler.transform(data_year[['Mes', 'Ano', 'Dia', 'Dia_da_semana', 'Trimestre']])
            predictions_year = best_rf.predict(data_year_scaled)
            data_year['Vacinas previstas'] = predictions_year
            data_year.reset_index(drop=True, inplace=True)  # Reset index to avoid index issues
            data_year.index += 1  # Start index at 1
            data_year.rename(columns={'Data': 'data', 'Vacinas aplicadas': 'vacinas_aplicadas', 'Vacinas previstas': 'vacinas_previstas'}, inplace=True)
            has_data = True

            # Ensure the directory for saving the graph exists
            media_dir = os.path.join(settings.BASE_DIR, 'suslog_project/static/images')
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Generate the graph
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data_year, x='data', y='vacinas_aplicadas', label='Real')
            sns.lineplot(data=data_year, x='data', y='vacinas_previstas', label='Predicted')
            plt.xticks(rotation=45)
            plt.title(f'Real vs Predicted Vaccines in {year_to_compare}')
            plt.xlabel('Date')
            plt.ylabel('Number of Vaccines')
            plt.legend()

            # Save the plot
            graph_filename = f'graph_{year_to_compare}.png'
            graph_path = os.path.join(media_dir, graph_filename)
            plt.savefig(graph_path)
            plt.close()

            # Update graph_path to be relative to the static directory
            graph_path = f'images/{graph_filename}'
            
    return render(request, 'predicao/index.html', {'form': form, 'data_year': data_year, 'has_data': has_data, 'graph_path': graph_path})
