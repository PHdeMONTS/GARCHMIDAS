import GarchMidas as gm
import pandas as pd
import datetime as dt

def load_macro_series(filename: str,seriesname: str)->pd.Series:
    mydf = pd.read_csv(filename, index_col=0, date_format='%Y-%m-%d')
    mydf.index = mydf.index.date
    mydf.columns=[seriesname]
    return mydf[seriesname]


if __name__ == '__main__':
    spx_series = load_macro_series("SPXdaily.csv",'SPXdaily')

    macro_data={}
    # US CPI
    CPI_series = load_macro_series("USCPI.csv",'USCPI')
    macro_data.update({'USCPI': (CPI_series,'M','US CPI All items seasonally adjusted','MIDAS',12)})

    # US M2/GDP
    M2GDP_series = load_macro_series("USM2GDPYoY.csv", 'USM2GDPYoY')
    macro_data.update({'USM2GDPYoY': (M2GDP_series, 'M', 'US M2 to GDP ratio YoY percentage change', 'Constrained', 12)})

    model = gm.GarchMidas('S&P 500 volatility', spx_series, macro_data,
                              sampleend=dt.date(year=2023, month=12, day=31))

    model.fit_GARCHMIDAS(niter=40, niter_omegas=20)

    print(model)
    model.simplechart_GARCHMIDAS(showrealised=False, maxvol=0.5)
    model.chart_filter()

