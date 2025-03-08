from functools import reduce

MEASURES_BY_CATEGORY = {
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
        'fiftyTwoWeekLow',
        'fiftyTwoWeekHigh',
        'fiftyTwoWeekLowChange',
        'fiftyTwoWeekLowChangePercent',
        'fiftyTwoWeekHighChange',
        'fiftyTwoWeekHighChangePercent',
        'fiftyTwoWeekRange',
        'fiftyTwoWeekChangePercent',
        'fiftyDayAverage',
        'twoHundredDayAverage',
        'fiftyDayAverageChange',
        'fiftyDayAverageChangePercent',
        'twoHundredDayAverageChange',
        'twoHundredDayAverageChangePercent'
    ],
    'Volume and Liquidity': [
        'volume',
        'regularMarketVolume',
        'averageVolume',
        'averageVolume10days',
        'averageDailyVolume10Day',
        'averageDailyVolume3Month'
    ],
    'Valuation Metrics': [
        'marketCap',
        'trailingPE',
        'forwardPE',
        'priceToSalesTrailing12Months',
        'priceToBook',
        'enterpriseValue',
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
        'returnOnAssets',
        'returnOnEquity',
        'profitMargins'
    ],
    'Dividends and Payouts': [
        'payoutRatio',
        'trailingAnnualDividendRate',
        'trailingAnnualDividendYield'
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
        'totalDebt',
    ]
}

ALL_MEASURES = reduce(list.__add__, [v for k, v in MEASURES_BY_CATEGORY.items()])