from functools import reduce

FA_INDICATORS_BY_CATEGORY = {
    'Price and Market Data': [
        'currentPrice',
        'previousClose',
        'open',
        'dayLow',
        'dayHigh',
        'regularMarketPreviousClose',
        'regularMarketOpen',
        'regularMarketDayLow',
        'regularMarketDayHigh',
        'regularMarketPrice',
        'regularMarketChange',
        'regularMarketChangePercent',
        'fiftyTwoWeekLow => Low_52weeks',
        'fiftyTwoWeekHigh => High_52weeks',
        'fiftyTwoWeekLowChange => ΔLow_52weeks',
        'fiftyTwoWeekLowChangePercent => ΔLow%_52weeks',
        'fiftyTwoWeekHighChange => ΔHigh_52weeks',
        'fiftyTwoWeekHighChangePercent => ΔHigh%_52weeks',
        'fiftyTwoWeekRange => Range_52weeks',
        'fiftyTwoWeekChangePercent => Δ52weeks',
        'fiftyDayAverage => MA50',
        'twoHundredDayAverage => MA250',
        'fiftyDayAverageChange => ΔMA50',
        'fiftyDayAverageChangePercent => ΔMA50%',
        'twoHundredDayAverageChange => ΔMA250',
        'twoHundredDayAverageChangePercent => ΔMA250%',
    ],
    'Volume and Liquidity': [
        'volume',
        'regularMarketVolume',
        'averageVolume',
        'averageVolume10days => avgVolume10d',
        'averageDailyVolume10Day => avgDailyVolume10d',
        'averageDailyVolume3Month => avgDailyVolume3M'
    ],
    'Valuation Metrics': [
        'marketCap',
        'trailingPE',
        'forwardPE',
        'priceToSalesTrailing12Months',
        'priceToBook => P/B',
        'enterpriseValue => EV',
        'enterpriseToRevenue',
        'enterpriseToEbitda',
        'trailingPegRatio'
    ],
    'Financial Performance': [
        'trailingEps',
        'forwardEps',
        'epsTrailingTwelveMonths',
        'epsForward',
        'epsCurrentYear',
        'priceEpsCurrentYear',
        'earningsQuarterlyGrowth',
        'netIncomeToCommon',
        'totalRevenue',
        'revenuePerShare',
        'grossProfits',
        'freeCashflow',
        'operatingCashflow',
        'earningsGrowth',
        'revenueGrowth',
        'grossMargins',
        'ebitdaMargins',
        'operatingMargins',
        'returnOnAssets => ROA',
        'returnOnEquity => ROE',
        'profitMargins'
    ],
    'Dividends and Payouts': [
        'payoutRatio',
        'trailingAnnualDividendRate',
        'trailingAnnualDividendYield => trailingDivYield'
    ],
    'Analyst Opinions and Targets': [
        'targetHighPrice',
        'targetLowPrice',
        'targetMeanPrice',
        'targetMedianPrice',
        'recommendationMean',
        'numberOfAnalystOpinions',
        'averageAnalystRating'
    ],
    'Short Interest and Float': [
        'sharesShort',
        'shortRatio',
        'shortPercentOfFloat',
        'floatShares',
        'sharesOutstanding'
    ],
    'Other Relevant Fields': [
        'beta',
        'quickRatio',
        'currentRatio',
        'debtToEquity',
        'totalCash',
        'totalCashPerShare',
        'ebitda',
        'totalDebt'
    ]
}

FA_INDICATORS = []
FA_INDICATORS_ALIAS = {}
for indicators in FA_INDICATORS_BY_CATEGORY.values():
    for indicator in indicators:
        parts = indicator.split("=>")
        ind = parts[0].strip()
        FA_INDICATORS.append(parts[0].strip())
        if len(parts) == 2:
            FA_INDICATORS_ALIAS[ind] = parts[1].strip()
