# Deploying Robo-AI-Book to Vercel

This document explains how to successfully deploy your Docusaurus-based Robo-AI-Book website to Vercel.

## Prerequisites

- Node.js >= 18.0 (as specified in your package.json)
- A Vercel account
- Your project pushed to a Git repository (GitHub, GitLab, or Bitbucket)

## Deployment Steps

### Method 1: Using Vercel CLI (Recommended for local testing)

1. Install Vercel CLI globally:
```bash
npm i -g vercel
```

2. Build your site locally to test:
```bash
npm run build
```

3. Deploy to Vercel:
```bash
vercel --prod
```

### Method 2: Connect Git Repository to Vercel

1. Push your code to a Git repository (if not already done)
2. Go to [Vercel Dashboard](https://vercel.com/dashboard)
3. Click "Add New..." > "Project"
4. Import your Git repository
5. Vercel should automatically detect this is a Docusaurus project
6. Use these build settings:
   - Build Command: `npm run build`
   - Output Directory: `build`
   - Install Command: `npm install`

## Configuration Files

### vercel.json
This file is crucial for Vercel deployment and is already configured in your project:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

### docusaurus.config.js
Updated to use root path deployment:
- baseUrl: '/'
- url: 'https://your-project-name.vercel.app'

## Troubleshooting Common Issues

1. **Build fails with "command not found"**
   - Ensure your package.json has the build script: `"build": "docusaurus build"`
   - Verify the output directory is set correctly in vercel.json

2. **Assets not loading after deployment**
   - Check that baseUrl is set to '/' for root deployment or the correct subdirectory
   - Verify all asset paths are relative

3. **Routing issues (404s on refresh)**
   - The vercel.json routes configuration handles this
   - The rewrite rule sends all traffic to index.html for client-side routing

## Environment Variables (if needed)

If you need environment variables, add them in the Vercel dashboard under:
Project Settings > Environment Variables

## Custom Domain Setup (Optional)

After successful deployment:
1. Go to your project in Vercel Dashboard
2. Navigate to Settings > Domains
3. Add your custom domain
4. Follow DNS configuration instructions

## Verification

After deployment, verify:
1. All pages load correctly
2. Navigation works properly
3. Assets (images, CSS, JS) load without 404s
4. Search functionality works (if implemented)
5. Internationalization works (your project supports English and Urdu)