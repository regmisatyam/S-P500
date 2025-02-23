import yfinance as yf

# Download S&P 500 data for the past year
sp500_data = yf.download('^GSPC', period='1y', interval='1d')

# Save data to CSV file
sp500_data.to_csv('sp500_data.csv')

print("Data downloaded and saved as sp500_data.csv")

