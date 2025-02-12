import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")


# Ver las primeras filas
df.head()

# Información general del dataset
df.info()

# Estadísticas descriptivas
df.describe()

# Ver cuántos valores nulos hay en cada columna
df.isnull().sum()

# Llenar valores nulos en la columna 'Age' con la mediana
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Llenar valores nulos en 'Embarked' con la moda
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Eliminar la columna 'Cabin' por demasiados valores nulos
df.drop(columns=['Cabin'], inplace=True)

# Contar cuántos sobrevivieron y cuántos no
sns.countplot(x='Survived', data=df, hue=None, palette='coolwarm', legend=False)
plt.title('Distribución de Supervivientes')
plt.show()

# Comparar tasa de supervivencia por género
sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title('Supervivencia por Género')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df, palette='Blues')
plt.title('Supervivencia por Clase de Boleto')
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(df['Age'], bins=30, kde=True, color='teal')
plt.title('Distribución de Edad de los Pasajeros')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=['number'])  # Solo columnas numéricas
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlaciones')
plt.show()
