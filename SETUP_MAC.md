# Trading Engine — Setup (Mac)

You downloaded ONE zip. Everything is already in the right folder structure
inside it. Do not move files around.

## 1. Unzip it
Double-click the zip in Finder. You'll get a folder called `trading-engine`
containing an `engine/` folder and this README. Keep it all together.

## 2. Open Terminal in that folder
Easiest way: in Finder, right-click the `trading-engine` folder →
"New Terminal at Folder". (Or open Terminal and `cd` into it.)

Verify you're in the right place — this must list an `engine` folder:
```
ls
```

## 3. Install dependencies (one time)
```
pip3 install -r requirements.txt
```
(Ignore any "pip version" warning — it's harmless.)

Confirm it worked:
```
python3 -c "import pandas, numpy, scipy, statsmodels; print('all good')"
```
You should see: all good

## 4. Set your FMP key (in the Terminal — there is no settings file)
Replace YOUR_KEY with your rotated FMP API key:
```
export FMP_API_KEY=YOUR_KEY
```
Nothing prints. That's normal — it's stored for this Terminal window.
(If you close Terminal, you have to run this line again next time.)

## 5. Run the session dissection on real TSLA
```
python3 -m engine.session TSLA --timeframe 5min --verbose
```
- `5min` is right for the Premium plan (1min is Ultimate-only).
- Drop `--date` to get the most recent session (simplest).
- To target a specific day: add `--date 2026-06-18`

## 6. Other things you can run
Gap statistics for a ticker:
```
python3 -m engine.gaps TSLA --lookback 10
```
Live reversal scanner:
```
python3 -m engine.intraday TSLA --timeframe 5min --verbose
```

## If something errors
Copy the WHOLE error and paste it back. The first real-data run may surface
something in how FMP's live JSON differs from expectations — that's expected,
and fixable fast.
