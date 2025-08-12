# 🚀 Deployment Guide - Indonesia Super League Football Analytics

This guide will help you deploy your Streamlit football analytics dashboard to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: You need a GitHub account to host your code
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Python Knowledge**: Basic understanding of Python and Git

## Step 1: Prepare Your Repository

### 1.1 Initialize Git Repository (if not already done)
```bash
git init
git add .
git commit -m "Initial commit: Indonesia Super League Football Analytics Dashboard"
```

### 1.2 Create GitHub Repository
1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `isuperleague-football-analytics`
3. Make it public (required for free Streamlit Cloud deployment)
4. Don't initialize with README (since you already have one)

### 1.3 Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign Up for Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Authorize Streamlit Cloud to access your repositories

### 2.2 Deploy Your App
1. Click **"New app"** button
2. Select your repository: `YOUR_USERNAME/YOUR_REPO_NAME`
3. Set the main file path: `app.py`
4. Set the app URL (optional): `isuperleague-football-analytics`
5. Click **"Deploy!"**

### 2.3 Wait for Deployment
- Streamlit Cloud will automatically install dependencies from `requirements.txt`
- The first deployment may take 2-3 minutes
- You'll see a progress bar and logs during deployment

## Step 3: Verify Deployment

### 3.1 Check Your App
- Once deployed, you'll get a URL like: `https://your-app-name.streamlit.app`
- Click the URL to verify your app is working
- Test all features: navigation, charts, data loading

### 3.2 Common Issues and Solutions

#### Issue: "Data file not found"
**Solution**: Ensure your `data/football_stats.csv` file is committed to GitHub
```bash
git add data/football_stats.csv
git commit -m "Add football statistics data"
git push
```

#### Issue: "Module not found"
**Solution**: Check your `requirements.txt` includes all dependencies
```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

#### Issue: "App not loading"
**Solution**: Check the deployment logs in Streamlit Cloud dashboard

## Step 4: Customize Your Deployment

### 4.1 Update App Settings
In Streamlit Cloud dashboard:
- **App URL**: Customize your app's URL
- **Repository**: Change source repository if needed
- **Main file path**: Update if you change the main file name

### 4.2 Environment Variables (if needed)
If your app needs environment variables:
1. Go to your app settings in Streamlit Cloud
2. Add environment variables under "Secrets"
3. Access them in your code using `st.secrets`

### 4.3 Continuous Deployment
- Streamlit Cloud automatically redeploys when you push changes to GitHub
- No manual intervention needed for updates

## Step 5: Share Your App

### 5.1 Public Sharing
- Your app URL is automatically public
- Share the URL with anyone: `https://your-app-name.streamlit.app`

### 5.2 Embedding
You can embed your app in websites:
```html
<iframe src="https://your-app-name.streamlit.app" width="100%" height="800"></iframe>
```

## Troubleshooting

### Common Deployment Issues

#### 1. **Import Errors**
- Check all imports in `app.py`
- Ensure all packages are in `requirements.txt`
- Test locally first: `streamlit run app.py`

#### 2. **Data Loading Issues**
- Verify file paths are relative to the app root
- Check file permissions and encoding
- Test data loading locally

#### 3. **Memory Issues**
- Optimize data loading with `@st.cache_data`
- Reduce data size if possible
- Use efficient data structures

#### 4. **Performance Issues**
- Enable caching for expensive operations
- Optimize chart rendering
- Use pagination for large datasets

### Getting Help

1. **Streamlit Cloud Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
2. **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
3. **GitHub Issues**: Report issues in your repository

## Advanced Configuration

### Custom Domain (Optional)
If you have a custom domain:
1. Go to app settings in Streamlit Cloud
2. Add your custom domain
3. Update DNS settings as instructed

### Multiple Environments
You can deploy multiple versions:
- **Production**: Main branch
- **Staging**: Development branch
- **Testing**: Feature branches

## Monitoring and Analytics

### App Analytics
Streamlit Cloud provides:
- Page views and user sessions
- Performance metrics
- Error logs and debugging info

### Performance Optimization
- Monitor app load times
- Optimize data processing
- Use efficient caching strategies

---

## Quick Deployment Checklist

- [ ] Code is working locally (`streamlit run app.py`)
- [ ] All files committed to GitHub
- [ ] `requirements.txt` is up to date
- [ ] `data/football_stats.csv` is included
- [ ] Repository is public
- [ ] Deployed to Streamlit Cloud
- [ ] App is accessible and functional
- [ ] All features tested
- [ ] App URL shared with stakeholders

---

**🎉 Congratulations!** Your Indonesia Super League Football Analytics Dashboard is now live and accessible to anyone with the URL.

**Next Steps:**
1. Share your app URL with your team
2. Monitor usage and performance
3. Plan regular data updates
4. Consider user feedback for improvements
