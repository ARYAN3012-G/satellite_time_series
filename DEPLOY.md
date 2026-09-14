# Deploying to Streamlit Community Cloud

## What ships in this repo (and why)

Only what `app/streamlit_app.py` needs at runtime:
- `app/streamlit_app.py` -- the app itself
- `src/` -- config, predictor, model architecture (imported by the app)
- `models/saved/drought_lstm.pt` (~215KB) + `drought_lstm_metadata.json` (~3KB)
- `requirements.txt`, `.streamlit/config.toml`

`data/raw/`, `data/processed/`, and `logs/` are excluded via `.gitignore` --
they're large, not needed to SERVE predictions (only to retrain), and
some of the raw data may not be something you want in a public repo.
If your GitHub repo needs to be public for the free Streamlit Cloud tier
and your data is sensitive, keep it private-repo instead (Streamlit
Cloud supports deploying from private repos too, with GitHub auth).

## Steps

1. **Push this repo to GitHub** (if not already):
   ```bash
   cd D:\satellite_time_series\maize_drought_detection
   git init                          # if not already a git repo
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```
   Double check `git status` doesn't show `data/raw/*` or `data/processed/*`
   about to be committed -- the .gitignore should already exclude them, but
   confirm before pushing, especially if this repo pre-dates the .gitignore.

2. **Go to https://share.streamlit.io** and sign in with GitHub.

3. **"New app"** -> select this repository, branch `main`,
   main file path: `app/streamlit_app.py`.

4. **Deploy.** First build takes a few minutes (installing torch is the
   slow part). Watch the build log for errors.

5. **If the torch install fails in the build log:** open
   `requirements.txt` in this repo, delete the `--extra-index-url` line
   and the `torch` line under "Try this first", and instead uncomment
   the plain `torch` line at the bottom. Commit, push -- Streamlit Cloud
   auto-redeploys on every push to the connected branch.

## After deploying

You'll get a URL like `https://<your-app-name>.streamlit.app` --
share this directly with anyone in your institute. No firewall rules,
no port forwarding, no client isolation concerns -- it's not on your
network at all.

## Updating the deployed model later

If you retrain (Module 8) and get a better model, just:
```bash
git add models/saved/
git commit -m "Updated model"
git push
```
Streamlit Cloud redeploys automatically.
