import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carregando o arquivo CSV
def explore_charts(path):
    df = pd.read_csv(path)

    # Convertendo as colunas para o tipo int64
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce').astype('Int64')
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce').astype('Int64')

    # Preenchendo valores ausentes com zero
    df['rank'].fillna(0, inplace=True)
    df['streams'].fillna(0, inplace=True)

    # Exibindo as primeiras linhas do DataFrame
    print(df.head())

    # Verificando as informações básicas do DataFrame
    print(df.info())

    # Realizando estatísticas descritivas das colunas numéricas
    print(df.describe())

    # Contando os valores únicos em cada coluna
    print(df.nunique())

    # Verificando a contagem de valores em uma coluna específica
    print(df['region'].value_counts())

    # Excluindo as colunas não numéricas
    numeric_columns = ['rank', 'streams']
    df_numeric = df[numeric_columns]

    # Verificando a correlação entre as colunas numéricas
    print(df_numeric.corr())

    # Gerando gráficos para visualização dos dados
    import matplotlib.pyplot as plt

    # Gráfico de barras para contar as ocorrências por região
    # df['region'].value_counts().plot(kind='bar')
    # plt.title('Contagem de músicas por região')
    # plt.xlabel('Região')
    # plt.ylabel('Contagem')
    # plt.show()

    # # Gráfico de dispersão para visualizar a relação entre rank e streams
    # plt.scatter(df['rank'], df['streams'])
    # plt.title('Rank vs Streams')
    # plt.xlabel('Rank')
    # plt.ylabel('Streams')
    # plt.show()
    return df

def predict_streams(df):
    # Carregar dados do arquivo CSV
    data = df

    # Separar os dados em recursos (X) e rótulos (y)
    X = data.drop(['streams'], axis=1)
    y = data['streams']

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

    categorical_cols = ['title', 'artist']
    # Criar o transformador para a codificação one-hot
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(sparse=False, drop='first'), categorical_cols)],
        remainder='passthrough'
    )

    # Aplicar a transformação one-hot aos dados de entrada de treinamento
    X_train_encoded = ct.fit_transform(X_train)

    # Aplicar a transformação one-hot aos dados de entrada de teste
    X_test_encoded = ct.transform(X_test)

    # Criar modelo de regressão linear
    model = LinearRegression()

    # Treinar o modelo com os dados de treinamento
    model.fit(X_train_encoded, y_train)

    # Fazer previsões com os dados de teste
    y_pred = model.predict(X_test_encoded)

    # Calcular o erro médio quadrático (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # Exemplo de previsão com novos dados
    new_data = pd.DataFrame({'nome da música': ['title'], 'nome do artista': ['artist'], 'data': ['2023-06-05']})

    # Aplicar a transformação one-hot aos novos dados
    new_data_encoded = ct.transform(new_data)

    # Fazer a previsão com os novos dados
    prediction = model.predict(new_data_encoded)
    print(f"Previsão de quantidade de streams: {prediction[0]}")
    

if __name__ == '__main__':
    path = 'C:/Users/Marcus/Documents/GitHub/ifal-songs-dwh/source/charts.csv'
    df = explore_charts(path)
    predict_streams(df)