import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
from pandas import DataFrame

def import_data_from_csv():
    df = pd.read_csv('src/Data_Task_MA.csv', sep=';')

    return df


def plot_data(df):
    fig = px.scatter(df, x="Year", y="Transport Demand")
    # fig.show()


def find_outliers_IQR(df):
    percentile25 = df['Transport Demand'].quantile(0.25)
    percentile75 = df['Transport Demand'].quantile(0.75)
    IQR = percentile75 - percentile25

    outliers = df['Transport Demand'][((df['Transport Demand'] <= (percentile25 - 1.5 * IQR)) |
                                       (df['Transport Demand'] >= (percentile75 + 1.5 * IQR)))]
    print("----------------------")
    print("Number of outliers: " + str(len(outliers)))
    print("Max outlier value: " + str(outliers.max()))
    print("Min outlier value: " + str(outliers.min()))

    print("----------------------")
    upper = np.where(df['Transport Demand'] >= (percentile75 + 1.5 * IQR))
    lower = np.where(df['Transport Demand'] <= (percentile25 - 1.5 * IQR))

    df.drop(upper[0], inplace=True) #inplace=True keyword in a pandas method changes the default behaviour
    df.drop(lower[0], inplace=True)

    return df


def linear_regression(df):
    years = df.drop('Transport Demand', axis='columns')
    transport_demands = df['Transport Demand']

    lr = linear_model.LinearRegression()
    lr.fit(years, transport_demands)
    predicted_transport_demands = lr.predict(years)

    # Linear regression line plot

    plt.figure(figsize=(12, 6))
    plt.scatter(df['Year'], df['Transport Demand'])
    plt.plot(df['Year'], predicted_transport_demands, 'r')
    plt.show()

    # original vs predicted plot

    plt.figure(figsize=(12, 6))
    plt.plot(df['Year'], df['Transport Demand'], label="Original")
    plt.plot(df['Year'], predicted_transport_demands, label="Predicted")
    plt.legend()
    plt.show()

    print("----------------------")
    print("Forecasted German transport demand in 2050: ", lr.predict([[2050]]))

    forecast_years = [2030, 2040, 2050]
    new_df = DataFrame(forecast_years, columns=['Forcasted Years'])
    forecast_transport_demand = lr.predict(new_df)
    new_df['Forecasted Transport Demand'] = forecast_transport_demand

    # plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(new_df['Forcasted Years'], new_df['Forecasted Transport Demand'])
    plt.xlabel('Years')
    plt.ylabel('Transport Demand')
    plt.title("Forecasted German Transport Demand")
    plt.show()

    # export to csv
    new_df.to_csv("output/Forecasted_Transport_Demand.csv")


if __name__ == '__main__':
    df = import_data_from_csv()
    print(df.describe()[['Year','Transport Demand']])
    print("Old Dataframe: ", df.shape)

    # plt.figure(figsize=(16, 8))
    # plt.subplot(2, 2, 1)
    # sns.histplot(df['Year'])
    # plt.subplot(2, 2, 2)
    # sns.boxplot(df['Transport Demand'])

    new_df = find_outliers_IQR(df)

    print(new_df.describe()[['Year','Transport Demand']])
    print("New Dataframe: ", new_df.shape)

    # plt.subplot(2, 2, 3)
    # sns.histplot(new_df['Year'])
    # plt.subplot(2, 2, 4)
    # sns.boxplot(new_df['Transport Demand'])
    # plt.show()

    linear_regression(df)