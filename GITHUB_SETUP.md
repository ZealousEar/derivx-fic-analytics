# GitHub Setup Guide

## Repository Name Recommendation
**`derivx-fic-analytics`**

## Step 1: Initialize Git (if not already done)
```bash
cd /Users/farhad/Code/Projects/Black-Scholes/release/v1
git init
```

## Step 2: Add all files
```bash
git add .
git commit -m "feat: DerivX v1.0 - FIC derivatives analytics platform

- Vectorized Black-Scholes with dividend/FX carry (q)
- SVI and SABR volatility smile calibration
- OIS/IRS term structure bootstrapping (PCHIP)
- Black IR caps/floors pricing with PV/DV01/PV01
- Streamlit UI (Equity/FX/Rates modes) with industrial design
- 8/8 unit tests passing (parity, implied vol, calibration)
- CI workflow + Jupyter notebooks + sample data"
```

## Step 3: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `derivx-fic-analytics`
3. Description: "FIC derivatives analytics: Black-Scholes, SVI/SABR, OIS/IRS bootstrapping, Black IR pricing, DV01/PV01"
4. Public repository
5. DO NOT initialize with README (we already have one)
6. Click "Create repository"

## Step 4: Link Remote & Push
```bash
# Replace <your-username> with your GitHub username
git remote add origin https://github.com/<your-username>/derivx-fic-analytics.git
git branch -M main
git push -u origin main
```

## Step 5: Create Release Tag
```bash
git tag -a v1.0 -m "Release v1.0: DerivX FIC Analytics MVP

Equity/FX/Rates modes with:
- Vectorized BS pricing & Greeks
- SVI/SABR calibration
- Term structure bootstrapping
- Black IR caps/floors
- PV/DV01/PV01 + scenario shocks
- All tests passing"

git push origin v1.0
```

## Step 6: Create GitHub Release (Optional but Recommended)
1. Go to https://github.com/<your-username>/derivx-fic-analytics/releases/new
2. Choose tag: `v1.0`
3. Release title: **DerivX v1.0 — FIC Derivatives Analytics**
4. Description:
```
Portfolio-grade derivatives analytics platform for Fixed Income & Currencies roles.

**Highlights:**
- Vectorized Black-Scholes with dividend/FX carry
- SVI (equity) and SABR (FX/rates) volatility calibration
- OIS/IRS curve bootstrapping with PCHIP fallback
- Black IR cap/floor pricing
- PV, DV01/PV01, parallel/twist scenario shocks
- Streamlit UI with industrial design (3 modes)
- 8/8 unit tests (parity, implied vol round-trip, calibration)
- CI workflow, Jupyter notebooks, sample data

**Tech:** Python 3.9+, NumPy, SciPy, Pandas, Plotly, Streamlit, Pytest
```
5. Click "Publish release"

## Optional: Add Topics/Tags on GitHub
Go to repository settings and add topics:
- `derivatives`
- `black-scholes`
- `fixed-income`
- `sabr`
- `svi`
- `python`
- `streamlit`
- `quantitative-finance`
- `risk-management`

---

## What's in this folder?
Everything needed for a recruiter-friendly GitHub repository:
- Clean README with screenshots
- All source code organized under `src/derivx/`
- Unit tests (8/8 passing)
- Jupyter notebooks (3 calibration demos)
- Sample data (rates quotes, equity chain)
- CI workflow (GitHub Actions)
- Procfile for deployment (Heroku/Render)

## Next Steps After Push
1. Verify CI passes on GitHub Actions
2. Add repository link to your resume/LinkedIn
3. Consider adding a GitHub repo description and website link

