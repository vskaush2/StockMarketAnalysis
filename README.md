# Cyclicity Analysis of Financial Time-Series Data
This repository contains a working implementation of Cyclicity Analysis, which is a pattern recognition technique for analyzing leader follower relationships amongst multiple time-series. We run Cyclicity Analysis on financial time-series pertaining to the stock and cryptocurrency markets. 

## Downloading Requirements
- Download PyCharm (free edition) from https://www.jetbrains.com/pycharm/.
- Download Python 3.10 from https://www.python.org/downloads/.

## Importing Project
- Open PyCharm and select `Get from VCS`.
- Enter this project's .git link.
- Specify the download location to be the location of your `PyCharmProjects` folder.
- Download the project.

## Installing Project Dependencies Automatically
- You may be prompted by PyCharm to install a Virtual Environment based on the `requirements.txt` file.
- Follow the onscreen instructions to do so.
- Make sure you specify your installed Python 3 for creating the Virtual Environment.


## Installing Project Dependencies Manually
- Open PyCharm Settings and locate the `Project: StockMarketAnalysis` pane.
- Click on `Project Interpreter`.
- Add a new `VirtualEnv` environment with your system Python.
- Restart PyCharm and open its local `Terminal`, which is located on the bottom of the PyCharm window.
- Type the command `pip3 install -r requirements.txt` in Terminal to install project dependencies.

## API Key
- In order to fetch time-series data, you **need** to get your own API key from https://polygon.io/. 
- For easy data fetching, it is strongly recommended you purchase the Starter plans in https://polygon.io/stocks#stocks-product-cards and https://polygon.io/crypto.
- Inside of the `FetchPrices.py`, replace the string 'xxxx' with your own API key string.

## Jupyter Server Instructions
  - Open the local `Terminal` on PyCharm.
  - Type the command `jupyter notebook` to open up a new Jupyter Server.
  - Click on the `CyclicityAnalysisDemo.ipynb` file to open the notebook.
  - Run each code cell using the toolbar on top of the window.
