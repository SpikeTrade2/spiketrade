# PythonAnywhere Deployment Guide

## Quick Start: Deploy SpikeTrade Strategy to PythonAnywhere

Follow these steps to deploy your trading strategy dashboard to PythonAnywhere.com

### Prerequisites
- A PythonAnywhere account (free tier works fine)
- Your app files (app.py)

---

## Step-by-Step Deployment

### 1. Create PythonAnywhere Account
1. Go to https://www.pythonanywhere.com
2. Sign up for a free account
3. Log in to your dashboard

### 2. Upload Your Application

**Option A: Upload via Web Interface**
1. Click on the "Files" tab
2. Create a new directory: `spiketrade`
3. Navigate into the directory
4. Click "Upload a file" and upload `app.py`

**Option B: Git Clone (Recommended)**
```bash
cd ~
git clone <your-repo-url> spiketrade
cd spiketrade
```

### 3. Install Dependencies

1. Open a **Bash console** from your PythonAnywhere dashboard
2. Run these commands:

```bash
cd ~/spiketrade
pip3.11 install --user streamlit plotly pandas numpy yfinance
```

Wait for installation to complete (2-3 minutes).

### 4. Create Streamlit Runner Script

Create a file called `run_streamlit.sh` in your app directory:

```bash
cd ~/spiketrade
nano run_streamlit.sh
```

Add this content:
```bash
#!/bin/bash
cd /home/yourusername/spiketrade
streamlit run app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true
```

Make it executable:
```bash
chmod +x run_streamlit.sh
```

### 5. Configure Web App

1. Go to the **Web** tab in PythonAnywhere
2. Click **"Add a new web app"**
3. Choose **"Manual configuration"** (not Django/Flask)
4. Select **Python 3.11**
5. Click Next

### 6. Edit WSGI Configuration

1. In the Web tab, find the "Code" section
2. Click on the WSGI configuration file link (it will look like `/var/www/yourusername_pythonanywhere_com_wsgi.py`)
3. **Delete all existing content** and replace with:

```python
import sys
import os

path = '/home/yourusername/spiketrade'  # Change 'yourusername' to your actual username
if path not in sys.path:
    sys.path.insert(0, path)

os.chdir(path)

from streamlit.web import cli as stcli
import sys

def application(environ, start_response):
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.port=8000",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    stcli.main()
```

**Important**: Replace `yourusername` with your actual PythonAnywhere username!

### 7. Create Streamlit Config

Create `.streamlit/config.toml`:

```bash
mkdir -p ~/spiketrade/.streamlit
nano ~/spiketrade/.streamlit/config.toml
```

Add this content:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 8000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### 8. Set Static Files (Optional)

In the Web tab, scroll to "Static files" section:
- URL: `/static/`
- Directory: `/home/yourusername/spiketrade/static/`

### 9. Reload Your Web App

1. Scroll to the top of the Web tab
2. Click the big green **"Reload"** button
3. Wait 10-30 seconds for the app to start

### 10. Access Your App

Your app will be available at:
```
https://yourusername.pythonanywhere.com
```

---

## Troubleshooting

### App Won't Load

**Check Error Logs:**
1. Go to Web tab
2. Scroll to "Log files" section
3. Check "Error log" and "Server log"

**Common Issues:**

1. **ImportError: No module named 'streamlit'**
   - Solution: Reinstall dependencies with `--user` flag:
     ```bash
     pip3.11 install --user streamlit plotly pandas numpy yfinance
     ```

2. **Wrong Python version**
   - Make sure you selected Python 3.11 when creating the web app
   - Check with: `python3.11 --version`

3. **Path issues**
   - Verify your username in WSGI file matches your actual PythonAnywhere username
   - Check paths: `/home/yourusername/spiketrade`

4. **Port already in use**
   - Each web app on PythonAnywhere gets its own port automatically
   - Use port 8000 in your config

### Slow Loading

- Free tier has limited resources
- First load may take 10-20 seconds
- Large datasets (1y+ of 1m data) will be slower

### Data Not Updating

- Yahoo Finance may rate-limit requests
- Try a different symbol
- Check internet connectivity in Bash console: `ping finance.yahoo.com`

---

## Performance Optimization

### For Free Tier

1. **Reduce Default Timeframe**
   - Use shorter periods (1mo, 3mo) instead of 1y
   - Use larger intervals (1d, 1h) instead of 1m

2. **Limit Prediction Lines**
   - Set max prediction lines to 20-50 instead of 100

3. **Cache Data** (Optional Enhancement)
   - Add `@st.cache_data` decorators to data fetching functions

---

## Updating Your App

When you make changes to `app.py`:

1. Upload new `app.py` via Files tab (or git pull if using Git)
2. Go to Web tab
3. Click **"Reload"** button
4. Wait 10-30 seconds
5. Refresh your browser

---

## Free vs Paid Accounts

**Free Account:**
- ✅ Perfect for personal use
- ✅ Sufficient for this app
- ❌ Slower performance
- ❌ Limited CPU time
- ❌ Single web app
- ❌ Auto-sleeps after inactivity

**Paid Account ($5/month):**
- ✅ Better performance
- ✅ More CPU time
- ✅ Multiple web apps
- ✅ Custom domains
- ✅ Always-on apps

---

## Alternative Deployment Options

If you need better performance or more features:

1. **Streamlit Cloud** - Free hosting for Streamlit apps
   - https://streamlit.io/cloud

2. **Heroku** - $7/month
   - Better performance than free PythonAnywhere

3. **AWS/Google Cloud** - Pay as you go
   - Best performance but more complex setup

4. **Replit** - Where you're currently working!
   - Can publish directly from Replit

---

## Security Notes

- This app uses Yahoo Finance public data (no API key required)
- No sensitive data is stored
- All calculations are done client-side in the browser
- Safe for public deployment

---

## Support

If you encounter issues:
1. Check PythonAnywhere forums: https://www.pythonanywhere.com/forums/
2. Check error logs in Web tab
3. Verify all dependencies are installed
4. Ensure WSGI file has correct username

---

## Next Steps After Deployment

1. **Test All Features**
   - Try different symbols (AAPL, MSFT, TSLA)
   - Test all timeframes
   - Adjust parameters to verify responsiveness

2. **Share Your App**
   - Share the URL: `https://yourusername.pythonanywhere.com`
   - Consider adding authentication if needed (Streamlit has built-in auth)

3. **Monitor Performance**
   - Check CPU usage in dashboard
   - Monitor load times
   - Optimize if needed

---

## Success Checklist

- [ ] PythonAnywhere account created
- [ ] app.py uploaded to ~/spiketrade
- [ ] Dependencies installed with `pip3.11 install --user`
- [ ] Web app created with Python 3.11
- [ ] WSGI file configured correctly
- [ ] .streamlit/config.toml created
- [ ] Web app reloaded
- [ ] App accessible at yourusername.pythonanywhere.com
- [ ] All features working (charts, indicators, signals)
- [ ] Different symbols tested successfully

---

**Congratulations!** 🎉 Your SpikeTrade Professional Strategy is now live on the internet!
