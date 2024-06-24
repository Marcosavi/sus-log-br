import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from .forms import ForquilhinhaVacinaForm, FuturePredictionForm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
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
    'min_samples_split': [2, 4, 8],
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
print("Melhores parametros encontrados: ", best_params)

# Train the model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train SVM with GridSearchCV
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]
}
svr = SVR()
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr,
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_svr.fit(X_train, y_train)
best_svr = grid_search_svr.best_estimator_
best_svr.fit(X_train, y_train)

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
            sns.lineplot(data=data_year, x='data', y='vacinas_previstas', label='Predito')
            plt.xticks(rotation=45)
            plt.title(f'Real vs Vacinas preditas em {year_to_compare}')
            plt.xlabel('Data')
            plt.ylabel('Número de vacinas')
            plt.legend()

            # Save the plot
            graph_filename = f'graph_{year_to_compare}.png'
            graph_path = os.path.join(media_dir, graph_filename)
            plt.savefig(graph_path)
            plt.close()

            # Update graph_path to be relative to the static directory
            graph_path = f'images/{graph_filename}'
            
    return render(request, 'predicao/passado.html', {'form': form, 'data_year': data_year, 'has_data': has_data, 'graph_path': graph_path})

def generate_future_dates(start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    future_df = pd.DataFrame({'Data': future_dates})
    future_df['Mes'] = future_df['Data'].dt.month
    future_df['Ano'] = future_df['Data'].dt.year
    future_df['Dia'] = future_df['Data'].dt.day
    future_df['Dia_da_semana'] = future_df['Data'].dt.dayofweek
    future_df['Trimestre'] = future_df['Data'].dt.quarter
    return future_df

def future_predictions(request):
    form = FuturePredictionForm(request.POST or None)
    future_df = pd.DataFrame()
    graph_path = None
    has_data = False
    if form.is_valid():
        year = form.cleaned_data['year']
        future_df = generate_future_dates(f'{year}-01-01', f'{year}-12-01')
        future_features = scaler.transform(future_df[['Mes', 'Ano', 'Dia', 'Dia_da_semana', 'Trimestre']])

        # Make predictions with trained models
        future_predictions_rf = best_rf.predict(future_features)
        future_predictions_lr = lr.predict(future_features)
        future_predictions_svr = best_svr.predict(future_features)

        # Add predictions to future_df
        future_df['Vacinas_RF'] = future_predictions_rf
        future_df['Vacinas_LR'] = future_predictions_lr
        future_df['Vacinas_SVR'] = future_predictions_svr
        has_data = True

        # Generate the graph
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=future_df, x='Data', y='Vacinas_RF', label='Random Forest')
        sns.lineplot(data=future_df, x='Data', y='Vacinas_LR', label='Linear Regression')
        sns.lineplot(data=future_df, x='Data', y='Vacinas_SVR', label='SVM')
        plt.xticks(rotation=45)
        plt.title(f'Future Predictions of Vaccine Numbers for {year}')
        plt.xlabel('Date')
        plt.ylabel('Number of Vaccines')
        plt.legend()

        # Save the plot
        graph_filename = f'future_predictions_{year}.png'
        media_dir = os.path.join(settings.BASE_DIR, 'suslog_project/static/images')
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)
        graph_path = os.path.join(media_dir, graph_filename)
        plt.savefig(graph_path)
        plt.close()

        # Update graph_path to be relative to the static directory
        graph_path = f'images/{graph_filename}'
    
    return render(request, 'predicao/futuro.html', {'form': form, 'future_df': future_df, 'has_data': has_data, 'graph_path': graph_path})
