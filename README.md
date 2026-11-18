# Momentum Dashboard

Monthly momentum picks from the S&P 500, with a backtest against SPY.

Live page: https://kimblejennings97.github.io/momentum-dashboard/

- `index.html` is the generated page that GitHub Pages serves.
- `picker/` holds the scoring script and page template.
- A GitHub Action (`.github/workflows/update-dashboard.yml`) reruns the picker on the 1st of each month and commits the new page. To update it any other time, use Actions → Update momentum dashboard → Run workflow.

Not investment advice.
